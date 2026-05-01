from __future__ import annotations
import argparse
import json
from dataclasses import dataclass
from pathlib import Path
import numpy as np
from mpi4py import MPI
import ufl
from dolfinx import fem, io, mesh as dmesh
from dolfinx.io import gmsh
from dolfinx.fem.petsc import LinearProblem

@dataclass(kw_only=True)
class Params:

    # Температуры, град. C
    T_init: float  # Начальная температура всего кота при t = 0
    T_blood: float # Температура крови
    T_battery: float  # Температура батареи на контактной поверхности
    T_env: float  # Температура внешней среды

    # Сколько тепла само выделяет тело внутри разных областей
    Q_tissue: float
    Q_heart: float
    Q_vessels: float

    # Насколько сильно кровь влияет на температуру ткани
    beta_tissue: float
    beta_heart: float
    beta_vessels: float

    # Теплообмен на границе, задаем тут коэффициенты теплообмена
    h_fur: float  # Обычная внешняя поверхность с шерстью
    h_extremity: float  # Лапы, уши, хвост, морда охлаждаются сильнее
    h_battery: float  # Контакт с батареей

    # Шаги по времени
    dt: float  # Шаг по времени
    t_end: float  # Общее время моделирования
    save_every: int  # Частота сохранения результата

    # Параметры материала по областям
    rho_tissue: float = 1050.0
    c_tissue: float = 3500.0
    k_tissue: float = 0.45

    rho_heart: float = 1060.0
    c_heart: float = 3600.0
    k_heart: float = 0.52

    rho_vessels: float = 1060.0
    c_vessels: float = 3800.0
    k_vessels: float = 0.70

    # Сосуды задаются как расчетная область около этих отрезков

    # Опорные точки сосудистой области в долях bounding box (огр. прямоугольник) кота.
    # Ячейки около соединяющих отрезков получают метку vessels.

    heart_center_frac: tuple[float, float, float] = (0.33, 0.50, 0.46)
    vessel_radius_frac: float

    aorta_arch_frac: tuple[float, float, float] = (0.40, 0.50, 0.63)
    pelvis_frac: tuple[float, float, float] = (0.62, 0.50, 0.55)

    front_left_shoulder_frac: tuple[float, float, float] = (0.22, 0.67, 0.46)
    front_right_shoulder_frac: tuple[float, float, float] = (0.22, 0.33, 0.46)
    front_left_paw_frac: tuple[float, float, float] = (0.11, 0.70, 0.10)
    front_right_paw_frac: tuple[float, float, float] = (0.11, 0.30, 0.10)

    back_left_hip_frac: tuple[float, float, float] = (0.65, 0.67, 0.44)
    back_right_hip_frac: tuple[float, float, float] = (0.65, 0.33, 0.44)
    back_left_paw_frac: tuple[float, float, float] = (0.77, 0.70, 0.10)
    back_right_paw_frac: tuple[float, float, float] = (0.77, 0.30, 0.10)

    # Кот лежит на батарее снизу: берем нижние фасеты в заданном x/y-диапазоне.
    battery_thickness_frac: float = 0.08  # Нижний слой для поиска контакта с батареей
    battery_x_range_frac: tuple[float, float] = (0.10, 0.80)  # Диапазон контакта по длине кота
    battery_y_range_frac: tuple[float, float] = (0.18, 0.82)  # Диапазон контакта по ширине кота

# Читает JSON-файл режима и превращает его в объект Params
def load_params(path: Path):
    data = json.loads(path.read_text(encoding="utf-8"))
    prm = Params(**data)
    prm.save_every = max(1, int(prm.save_every))  # Минимум 1, чтобы не было step % 0.
    return prm

# По имени режима находит соответствующий JSON-файл в папке modes/
def resolve_mode_path(mode: str):
    path = Path(__file__).resolve().parent / "modes" / f"{mode}.json"
    if not path.exists():
        raise FileNotFoundError(f"Режим '{mode}' не найден: {path}")
    return path

# Функция чтения сетки кота из .msh файла Gmsh
def read_mesh(msh_file: Path):
    meshdata = gmsh.read_from_msh(str(msh_file), MPI.COMM_WORLD, gdim=3)
    return meshdata.mesh, meshdata.cell_tags, getattr(meshdata, "physical_groups", {})

# При параллельном запуске каждый процесс видит только свою часть сетки
# эта функция суммирует local_count по всем процессам
def global_cnt(msh, local_count: int):
    return int(msh.comm.allreduce(int(local_count), op=MPI.SUM))

# Считаем минимальный прямоугольный 3D-параллелепипед, в который помещается весь кот
def global_box(msh):
    X = msh.geometry.x  # Массив всех координат узлов сетки

    bmin_local = np.min(X, axis=0)  # Локальный минимум по всем координатам
    bmax_local = np.max(X, axis=0)  # Локальный максимум по всем координатам

    # DOLFINX удобен тем, что можно решать задачи быстрее на параллельных процессах
    # В моей текущей реализации задача не такая сложная, можно было в целом обойтись без добавения процессов
    # При параллельном запуске, у каждого процесса свой кусок сетки, поэтому ищем минимум по всем процессам

    bmin = np.array([msh.comm.allreduce(float(v), op=MPI.MIN) for v in bmin_local])
    bmax = np.array([msh.comm.allreduce(float(v), op=MPI.MAX) for v in bmax_local])
    return bmin, bmax

# Функция перевода относительных долей в реальные координаты
def frac_to_xyz(bmin: np.ndarray, bmax: np.ndarray, frac):
    frac = np.array(tuple(frac), dtype=np.float64)
    return bmin + frac * (bmax - bmin)

# Функция для печати информации
def report_tags(msh, cell_tags, facet_tags, tags) -> None:
    HEART, VESSELS, TISSUE, BATTERY, FUR, EXTREMITY = tags
    n_heart = global_cnt(msh, len(cell_tags.find(HEART)))
    n_vessels = global_cnt(msh, len(cell_tags.find(VESSELS)))
    n_tissue = global_cnt(msh, len(cell_tags.find(TISSUE)))
    n_battery = global_cnt(msh, len(facet_tags.find(BATTERY)))
    n_fur = global_cnt(msh, len(facet_tags.find(FUR)))
    n_extremity = global_cnt(msh, len(facet_tags.find(EXTREMITY)))
    if msh.comm.rank == 0:
        print("Ячейки областей: " f"сердце={n_heart}, сосуды={n_vessels}, ткани={n_tissue}")
        print("Граничные фасеты: " f"батарея={n_battery}, шерсть={n_fur}, тонкие части={n_extremity}")

# Квадраты расстояний от точек до отрезка a-b
# Нужны, чтобы пометить ячейки рядом с сосудистой линией
def point_segment_distance2(points: np.ndarray, a: np.ndarray, b: np.ndarray):
    ab = b - a  # Вектор от начала отрезка к концу
    denom = float(np.dot(ab, ab))  # Длина отрезка в квадрате
    if denom < 1.0e-20:
        return np.sum((points - a) ** 2, axis=1)
    ap = points - a
    tau = np.sum(ap * ab, axis=1) / denom
    tau = np.clip(tau, 0.0, 1.0)
    proj = a + tau[:, None] * ab[None, :]
    return np.sum((points - proj) ** 2, axis=1)

# Функция определения тетраэдров для сосудистой области
def make_vessel_mask(mids: np.ndarray, bmin: np.ndarray, bmax: np.ndarray, prm: Params):
    vessel_segments = make_vessel_segments(bmin, bmax, prm)  # Строим список сосудистых отрезков
    r = float(prm.vessel_radius_frac) * float(np.min(bmax - bmin))  # Задаем радиус сосуда
    is_vessel = np.zeros(mids.shape[0], dtype=bool)
    for a, b in vessel_segments: 
        is_vessel |= point_segment_distance2(mids, a, b) <= r * r
    return is_vessel  # True для тетраэдров, попавших в сосудистую область

# Построение списка сосудистых отрезков.
def make_vessel_segments(bmin: np.ndarray, bmax: np.ndarray, prm: Params):

    p_heart = frac_to_xyz(bmin, bmax, prm.heart_center_frac)
    p_arch = frac_to_xyz(bmin, bmax, prm.aorta_arch_frac)
    p_pelvis = frac_to_xyz(bmin, bmax, prm.pelvis_frac)

    # Передние лапы
    p_fl_sh = frac_to_xyz(bmin, bmax, prm.front_left_shoulder_frac)
    p_fr_sh = frac_to_xyz(bmin, bmax, prm.front_right_shoulder_frac)
    p_fl_paw = frac_to_xyz(bmin, bmax, prm.front_left_paw_frac)
    p_fr_paw = frac_to_xyz(bmin, bmax, prm.front_right_paw_frac)

    # Задние лапы
    p_back_left_hip = frac_to_xyz(bmin, bmax, prm.back_left_hip_frac)
    p_back_right_hip = frac_to_xyz(bmin, bmax, prm.back_right_hip_frac)
    p_back_left_paw = frac_to_xyz(bmin, bmax, prm.back_left_paw_frac)
    p_back_right_paw = frac_to_xyz(bmin, bmax, prm.back_right_paw_frac)

    return [(p_heart, p_arch), (p_arch, p_pelvis), (p_heart, p_fl_sh), (p_fl_sh, p_fl_paw), (p_heart, p_fr_sh), (p_fr_sh, p_fr_paw),
        (p_pelvis, p_back_left_hip), (p_back_left_hip, p_back_left_paw), (p_pelvis, p_back_right_hip), (p_back_right_hip, p_back_right_paw)]

def get_physical_tag(physical_groups: dict, dim: int, name: str):
    pg = physical_groups.get(name)
    if pg is not None and pg.dim == dim:
        return int(pg.tag)
    return None

# Функция определяет, какие граничные фасеты относятся к тонким/крайним частям тела
# Именно для них будем применять коэффициент h_extremity
def classify_extremity_facets(mids: np.ndarray, bmin: np.ndarray, bmax: np.ndarray):
    L = np.maximum(bmax - bmin, 1.0e-12)
    frac = (mids - bmin[None, :]) / L[None, :] # Перевод в доли

    # Доли по каждой оси
    x = frac[:, 0]
    y = frac[:, 1]
    z = frac[:, 2]

    # Тонкие части охлаждаются сильнее: голова/морда, уши, лапы, хвост.
    head_or_muzzle = x < 0.22
    ears = (x < 0.32) & (z > 0.72)
    paws = z < 0.22
    tail = x > 0.82
    side_legs = (z < 0.45) & (np.abs(y - 0.5) > 0.30)
    return head_or_muzzle | ears | paws | tail | side_legs

# Функция определения принадлежности тетраэдров конкретной категории
def make_anatomical_cell_tags(msh, cell_tags_in, physical_groups, prm: Params):
    tdim = msh.topology.dim  # tdim = 3
    msh.topology.create_connectivity(tdim, tdim)
    num_cells_local = msh.topology.index_map(tdim).size_local
    cells = np.arange(num_cells_local, dtype=np.int32) # Массив индексов ячеек

    HEART, VESSELS, TISSUE = 1, 2, 3
    exact_heart = get_physical_tag(physical_groups, 3, "heart")
    exact_tiss = get_physical_tag(physical_groups, 3, "tissue")

    values = np.full(cells.shape, TISSUE, dtype=np.int32) # Массив меток для всех локальных ячеек

    for idx, val in zip(cell_tags_in.indices, cell_tags_in.values):
        if int(idx) >= num_cells_local:
            continue
        if int(val) == exact_heart:
            values[int(idx)] = HEART
        elif int(val) == exact_tiss:
            values[int(idx)] = TISSUE

    mids = dmesh.compute_midpoints(msh, tdim, cells)  # Считаем центы всех тетраэдров
    bmin, bmax = global_box(msh)
    is_vessel = make_vessel_mask(mids, bmin, bmax, prm)
    values[is_vessel & (values != HEART)] = VESSELS
    cell_tags = dmesh.meshtags(msh, tdim, cells, values)
    return cell_tags, HEART, VESSELS, TISSUE

# Функция классификации поверхности на 3 типа:
# BATTERY - контакт с батареей, FUR - обычная поверхность с шерстью, EXTREMITY - тонкие части: лапы, уши, хвост, морда
def make_boundary_facet_tags(msh, prm: Params):
    BATTERY, FUR, EXTREMITY = 10, 20, 30

    tdim = msh.topology.dim
    fdim = tdim - 1
    msh.topology.create_connectivity(fdim, tdim)
    boundary_facets = np.asarray(dmesh.exterior_facet_indices(msh.topology), dtype=np.int32)  # Массив номеров всех поверхностных треугольников кота

    mids = dmesh.compute_midpoints(msh, fdim, boundary_facets)
    bmin, bmax = global_box(msh)
    L = bmax - bmin

    x0 = bmin[0] + prm.battery_x_range_frac[0] * L[0]
    x1 = bmin[0] + prm.battery_x_range_frac[1] * L[0]

    y0 = bmin[1] + prm.battery_y_range_frac[0] * L[1]
    y1 = bmin[1] + prm.battery_y_range_frac[1] * L[1]

    z_thr = bmin[2] + prm.battery_thickness_frac * L[2]

    is_battery = ((mids[:, 2] <= z_thr) & (mids[:, 0] >= x0) & (mids[:, 0] <= x1) & (mids[:, 1] >= y0) & (mids[:, 1] <= y1))
    is_extremity = classify_extremity_facets(mids, bmin, bmax)

    values = np.full(boundary_facets.shape, FUR, dtype=np.int32)
    values[is_extremity] = EXTREMITY
    values[is_battery] = BATTERY

    facet_tags = dmesh.meshtags(msh, fdim, boundary_facets, values)
    return facet_tags, BATTERY, FUR, EXTREMITY

# Задаем разные физические параметры в разных областях кота
def make_region_field(msh, cell_tags, tag_values: dict[int, float], default_value: float | None = None, name: str = ""):
    V0 = fem.functionspace(msh, ("DG", 0))  # Создаем пространство конечных элементов на сетке msh
    f = fem.Function(V0)  # Создаем функцию в этом пространстве
    if default_value is None:
        default_value = next(iter(tag_values.values()))
    f.x.array[:] = float(default_value)
    for tag, value in tag_values.items():
        cells = cell_tags.find(tag)
        f.x.array[cells] = float(value)
    if name:
        f.name = name
    return f

# Записывает region_id, k, beta и Q в отдельный XDMF-файл, чтобы можно было посмотреть области и параметры модели в ParaView
def write_auxiliary_fields(msh, outdir: Path, mode_name: str, region_id, k, beta, Q):
    aux_path = outdir / f"auxiliary_fields_{mode_name}.xdmf"
    with io.XDMFFile(msh.comm, str(aux_path), "w") as xdmf:
        xdmf.write_mesh(msh)
        xdmf.write_function(region_id, 0.0)
        xdmf.write_function(k, 0.0)
        xdmf.write_function(beta, 0.0)
        xdmf.write_function(Q, 0.0)
    return aux_path

# Основной расчет: собирает коэффициенты, задает слабую форму уавнения, решает задачу по временным шагам и записывает температуру в XDMF.
def solve_transient(msh, cell_tags, facet_tags, tags, prm: Params, outdir: Path, mode_name: str):
    HEART, VESSELS, TISSUE, BATTERY, FUR, EXTREMITY = tags

    # Берём коэффициенты теплообмена из параметров режима и приводим к float
    h_fur = float(prm.h_fur)
    h_extremity = float(prm.h_extremity)
    h_battery = float(prm.h_battery)

    # Печатаем крткое описание запуска только на главном MPI-процессе, чтобы при параллельном запуске сообщение не дублировалось
    if msh.comm.rank == 0:
        print(
            f"Режим {mode_name}: h_fur={h_fur}, h_extremity={h_extremity}, h_battery={h_battery}\n"
            f"Температуры: T_init={prm.T_init}, T_blood={prm.T_blood}, "
            f"T_battery={prm.T_battery}, T_env={prm.T_env} град. C\n"
            f"Время: dt={prm.dt} с, t_end={prm.t_end} с, save_every={prm.save_every}"
        )

    # Создаём пространство непрерывных линейных конечных элементов для температуры
    V = fem.functionspace(msh, ("Lagrange", 1))

    # T — пробная функция, т.е. неизвестная температура, которую мы ищем на новом шаге времени
    T = ufl.TrialFunction(V)

    # v — тестовая функция, она нужна для слабой формы метода конечых элементов
    v = ufl.TestFunction(V)

    # Создаем кусочно-постоянные поля коэффициентов: в каждой ячейке значение выбираеся по ее области HEART/VESSELS/TISSUE.
    rho = make_region_field(msh, cell_tags, {HEART: prm.rho_heart, VESSELS: prm.rho_vessels, TISSUE: prm.rho_tissue}, name="rho")
    c = make_region_field(msh, cell_tags, {HEART: prm.c_heart, VESSELS: prm.c_vessels, TISSUE: prm.c_tissue}, name="c")
    k = make_region_field(msh, cell_tags, {HEART: prm.k_heart, VESSELS: prm.k_vessels, TISSUE: prm.k_tissue}, name="k")
    Q = make_region_field(msh, cell_tags, {HEART: prm.Q_heart, VESSELS: prm.Q_vessels, TISSUE: prm.Q_tissue}, name="Q_internal")
    beta = make_region_field(msh, cell_tags, {HEART: prm.beta_heart, VESSELS: prm.beta_vessels, TISSUE: prm.beta_tissue}, name="beta_perf")
    region_id = make_region_field(msh, cell_tags, {HEART: float(HEART), VESSELS: float(VESSELS), TISSUE: float(TISSUE)}, name="region_id")

    # Подготавливаем dt, T_blood, T_env и T_battery для математической формы уравнения
    dt = fem.Constant(msh, float(prm.dt))
    T_blood = fem.Constant(msh, float(prm.T_blood))
    T_env = fem.Constant(msh, float(prm.T_env))
    T_bat = fem.Constant(msh, float(prm.T_battery))

    # Меры интегрирования для слабой формы:
    # dx интегрирует по объему с учетом cell_tags,
    # ds интегрирует по поверхности с учетом facet_tags.
    dx = ufl.Measure("dx", domain=msh, subdomain_data=cell_tags)
    ds = ufl.Measure("ds", domain=msh, subdomain_data=facet_tags)

    # Tn — температура на предыдущем временном шаге
    Tn = fem.Function(V)
    Tn.name = "Temperature"

    # Заполняем всю начальную температуру одним значением (во всех узлах сетки)
    Tn.x.array[:] = float(prm.T_init)

    # В моем случае мы скорее будем запускать на одном процессе, поэтому можно убрать эту проверку
    # Однако если используем несколько MPI-процессов, то эта проверка нужна для синхронизации общих значений функции
    if hasattr(Tn.x, "scatter_forward"):
        Tn.x.scatter_forward()

    # Создаём коэффициент при временной производной.
    mass = rho * c / dt

    # Неявная схема Эйлера:
    # левая часть содержит неизвестную T на новом шаге,
    # правая часть содержит температуру предыдущего шага Tn.
    a = (
        mass * T * v * dx
        + ufl.inner(k * ufl.grad(T), ufl.grad(v)) * dx
        + beta * T * v * dx
        + float(h_fur) * T * v * ds(FUR)
        + float(h_extremity) * T * v * ds(EXTREMITY)
        + float(h_battery) * T * v * ds(BATTERY)
    )

    L = (
        mass * Tn * v * dx
        + (Q + beta * T_blood) * v * dx
        + float(h_fur) * T_env * v * ds(FUR)
        + float(h_extremity) * T_env * v * ds(EXTREMITY)
        + float(h_battery) * T_bat * v * ds(BATTERY)
    )

    problem = LinearProblem(a, L,
        petsc_options_prefix=f"cat_bioheat_{mode_name}_",
        petsc_options={
            "ksp_type": "preonly",
            "pc_type": "lu"
        }
    )

    # Создаём папку для результатов
    outdir.mkdir(parents=True, exist_ok=True)

    # Сохраняем вспомогательные поля
    aux_path = write_auxiliary_fields(msh, outdir, mode_name, region_id, k, beta, Q)

    # Создаём путь к файлу, куда потом будет записываться температура во времени
    temp_path = outdir / f"temperature_{mode_name}.xdmf"

    t = 0.0
    nsteps = int(np.ceil(prm.t_end / prm.dt))

    with io.XDMFFile(msh.comm, str(temp_path), "w") as xdmf:
        xdmf.write_mesh(msh)
        xdmf.write_function(Tn, t)

        # Запускаем цикл по временным шагам
        for step in range(1, nsteps + 1):
            t += prm.dt
            Th = problem.solve()
            Th.name = "Temperature"

            if step % prm.save_every == 0 or step == nsteps:
                xdmf.write_function(Th, t)

            # Обновляем старую температуру
            Tn.x.array[:] = Th.x.array

            # Синхронизируем значения между MPI-процессами после ручной записи в массив
            # Для одного процесса почти не важно, для MPI-запуска полезно
            if hasattr(Tn.x, "scatter_forward"):
                Tn.x.scatter_forward()
    return temp_path, aux_path, Tn

# Считает глобальные минимум и максимум температуры по всей сетке.
def report_temperature_range(msh, T):
    Tmin_local = float(np.min(T.x.array))
    Tmax_local = float(np.max(T.x.array))
    Tmin = msh.comm.allreduce(Tmin_local, op=MPI.MIN)
    Tmax = msh.comm.allreduce(Tmax_local, op=MPI.MAX)
    return Tmin, Tmax

# Собирает интеграл в одно число и суммирует вклад по всем MPI-процессам
# Нужна для подсчета средней температуры по области
def assemble_global_scalar(msh, expr):
    local = fem.assemble_scalar(fem.form(expr))
    return float(msh.comm.allreduce(local, op=MPI.SUM))

# Считает и печатает средние финальные температуры по всему коту и по области
def report_final_temperature_summary(msh, T, cell_tags, tags):
    HEART, VESSELS, TISSUE, *_ = tags
    dx = ufl.Measure("dx", domain=msh, subdomain_data=cell_tags)
    
    # Cчитает среднюю температуру на области с меткой tag
    def mean_on(tag: int):
        vol = assemble_global_scalar(msh, 1.0 * dx(tag))
        if vol <= 0.0:
            return float("nan")
        heat = assemble_global_scalar(msh, T * dx(tag))
        return heat / vol

    total_vol = assemble_global_scalar(msh, 1.0 * dx) # Общий объём всего кота
    total_mean = assemble_global_scalar(msh, T * dx) / total_vol # Средняя температура по всему коту

    # Средние температуры по отдельным областям
    heart_mean = mean_on(HEART) 
    vessels_mean = mean_on(VESSELS)
    tissue_mean = mean_on(TISSUE)

    if msh.comm.rank != 0:
        return

    print(
        "Итоговые средние температуры: "
        f"весь кот={total_mean:.3f}, ткани={tissue_mean:.3f}, "
        f"сердце={heart_mean:.3f}, сосуды={vessels_mean:.3f} град. C"
    )

def main():
    # Создаем парсер аргументов командной строки
    ap = argparse.ArgumentParser()

    # Обязательный путь к входной сетке Gmsh (.msh)
    ap.add_argument("--mesh", required=True, help="путь к входной сетке .msh")

    # Обязательное имя режима из папки modes/
    ap.add_argument("--mode", required=True, help="имя JSON-режима из папки modes рядом со скриптом, например room_radiator")

    # Папка для результатов; если не указана, используется out
    ap.add_argument("--out", default="out", help="папка для результатов")

    # Читаем аргументы командной строки
    args = ap.parse_args()

    # По имени режима находим соответствующий JSON-файл
    params_path = resolve_mode_path(args.mode)

    # Имя режима без расширения .json
    mode_name = params_path.stem

    # Загружаем параметры расчета из JSON
    prm = load_params(params_path)

    if MPI.COMM_WORLD.rank == 0:
        print(f"Режим: {mode_name}")
        print(f"Файл параметров: {params_path}")

    msh, cell_tags_in, physical_groups = read_mesh(Path(args.mesh))

    # Размечаем объемные ячейки: сердце, сосуды, ткани
    cell_tags, HEART, VESSELS, TISSUE = make_anatomical_cell_tags(msh, cell_tags_in, physical_groups, prm)

    # Размечаем поверхность: батарея, шерсть, тонкие части тела
    facet_tags, BATTERY, FUR, EXTREMITY = make_boundary_facet_tags(msh, prm)

    # Собираем все числовые метки в один кортеж для передачи в функции
    tags = (HEART, VESSELS, TISSUE, BATTERY, FUR, EXTREMITY)

    # Печатаем количество ячеек/фасетов в каждой области
    report_tags(msh, cell_tags, facet_tags, tags)

    # Запускаем нестационарный расчет температуры.
    temp_path, aux_path, Tfinal = solve_transient(
        msh, cell_tags, facet_tags, tags, prm, Path(args.out), mode_name
    )

    # Считаем минимальную и максимальную финальную температуру
    Tmin, Tmax = report_temperature_range(msh, Tfinal)

    # Печатаем средние финальные температуры по областям
    report_final_temperature_summary(msh, Tfinal, cell_tags, tags)

    # Печатаем итоговые файлы и диапазон температур только главным процессом
    if msh.comm.rank == 0:
        print(f"[{mode_name}] файл температуры: {temp_path}")
        print(f"[{mode_name}] вспомогательные поля: {aux_path}")
        print(f"[{mode_name}] итоговые Tmin/Tmax = {Tmin:.3f} / {Tmax:.3f} град. C")

if __name__ == "__main__":
    main()
