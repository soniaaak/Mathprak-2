#include <gmsh.h>
#include <algorithm>
#include <cmath>
#include <iostream>
#include <set>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

/* ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ */

// Вводим удобное обозначение, тк дальше будем часто пользоваться парой (dimension, tag) - так в Gmsh геом.сущности представляются
using DimTag = std::pair<int, int>;

// Констанста пи для поворотов 
static constexpr double PI = 3.14159265358979323846;

// Структура для хранения вектора
struct Vec3 {
    double x = 0.0;
    double y = 0.0;
    double z = 0.0;
};

// Разность векторов
static Vec3 operator-(const Vec3 &a, const Vec3 &b) {
    return Vec3{a.x - b.x, a.y - b.y, a.z - b.z};
}

// Умнодение вектора на константу
static Vec3 operator*(double s, const Vec3 &v) {
    return Vec3{s * v.x, s * v.y, s * v.z};
}

// Пересчет угла из градусов в радианы
static double deg(double a) {
    return a * PI / 180.0;
}

// Возвращает тег первого найденного объема
static int firstVolume(const std::vector<DimTag> &out) {
    for(const auto &e : out) {
        if(e.first == 3) return e.second;
    }
    throw std::runtime_error("Не найден 3D-объем");
}

// Поворот геометрических объектов dt вокруг осей x, y, z
// c.x, c.y, c.z — точка центра поворота, ang - угол поворота в радианах
static void rotateXYZ(const std::vector<DimTag> &dt, const Vec3 &c, const Vec3 &ang) {
    if (std::abs(ang.x) > 1e-14) {
        gmsh::model::occ::rotate(dt, c.x, c.y, c.z, 1, 0, 0, ang.x);
    }
    if(std::abs(ang.y) > 1e-14) {
        gmsh::model::occ::rotate(dt, c.x, c.y, c.z, 0, 1, 0, ang.y);
    }
    if(std::abs(ang.z) > 1e-14) {
        gmsh::model::occ::rotate(dt, c.x, c.y, c.z, 0, 0, 1, ang.z);
    }
}

// Построение эллипсоида
// с - центр эллипсоида, r - коэф.растяжения по трем осям, angRad - углы поворота по осям в радианах
static int addEllipsoid(const Vec3 &c, const Vec3 &r, const Vec3 &ang) {
    int tag = gmsh::model::occ::addSphere(c.x, c.y, c.z, 1.0);
    std::vector<DimTag> dt = {{3, tag}};
    gmsh::model::occ::dilate(dt, c.x, c.y, c.z, r.x, r.y, r.z);
    rotateXYZ(dt, c, ang);
    return tag;
}

// Построение цилиндра с округленными концами (как капсула)
// Берем две точки, между ними строим цилиндр радуиса r, на концах добавялем сферы того же радуиса, соединяем все это в один объем
static int addCapsule(const Vec3 &a, const Vec3 &b, double r) {
    Vec3 d = b - a;
    int cyl = gmsh::model::occ::addCylinder(a.x, a.y, a.z, d.x, d.y, d.z, r);
    int s1  = gmsh::model::occ::addSphere(a.x, a.y, a.z, r);
    int s2  = gmsh::model::occ::addSphere(b.x, b.y, b.z, r);
    std::vector<DimTag> out;
    std::vector<std::vector<DimTag>> outMap;
    gmsh::model::occ::fuse({{3, cyl}}, {{3, s1}, {3, s2}}, out, outMap, -1, true, true);
    return firstVolume(out);
}

// Склейка нескольких объемов в один
// Принимает на вход список тегов объемов, возвращает тег объединённого объёма
static int fuseList(const std::vector<int> &vols) {
    if(vols.size() == 1) {
        return vols[0];
    }
    std::vector<DimTag> obj = {{3, vols[0]}};

    std::vector<DimTag> news;
    for(std::size_t i = 1; i < vols.size(); ++i) {
        news.push_back({3, vols[i]});

    }

    std::vector<DimTag> out;
    std::vector<std::vector<DimTag>> outMap;
    gmsh::model::occ::fuse(obj, news, out, outMap, -1, true, true);
    return firstVolume(out);
}

// Настройка сетки
static void configureMesh(double lcMin, double lcMax) {
    gmsh::option::setNumber("General.Terminal", 1); // Вывод сообщений в терминал

    gmsh::option::setNumber("Mesh.MeshSizeMin", lcMin); // Минимальный размер сетки, мельче него нельзя
    gmsh::option::setNumber("Mesh.MeshSizeMax", lcMax); // Максимальный размер сетки, крупнее него нельзя

    gmsh::option::setNumber("Mesh.MeshSizeFromPoints", 0); // Отключение влияния точек на размер сетки (чтобы не было сгущение около отдельных точек)
    gmsh::option::setNumber("Mesh.MeshSizeFromCurvature", 20); // Сгущение сетки по кривизне поверхности, в более изогнутых местах сетка будет мельче
    gmsh::option::setNumber("Mesh.MeshSizeExtendFromBoundary", 0); // Отключение распространения размеров с границы внутрь

    gmsh::option::setNumber("Mesh.Algorithm3D", 1); // Delaunay устойчивее для внутренних разбиений
    gmsh::option::setNumber("Mesh.Optimize", 1); // Улучшаем качество готовой сетки
    gmsh::option::setNumber("Mesh.OptimizeNetgen", 1); // Улучшаем качество готовой сетки

    gmsh::option::setNumber("Geometry.OCCFixDegenerated", 1); // Исправляем вырожденные элементы геометрии
    gmsh::option::setNumber("Geometry.OCCFixSmallEdges", 1); // Убираем проблемы с мелкими ребрами
    gmsh::option::setNumber("Geometry.OCCFixSmallFaces", 1); // Чиним слишком маленькие грани
    gmsh::option::setNumber("Geometry.OCCSewFaces", 1); // Сшиваем соседние поверхности
    gmsh::option::setNumber("Geometry.OCCMakeSolids", 1); // Собираем замкнутые оболочки в объемы
}

/* ВСПОМОГАТЕЛЬНЫЕ СТРУКТУРЫ */

struct EllipsoidSpec {
    Vec3 c; // Центр эллипсиода
    Vec3 r; // Радиусы эллипсоида по трем осям
};

struct ThermalRegions {
    EllipsoidSpec heart; // Сердце в виде эллипсиода
};

// Структура для хранения групп объёмов в модели
struct VolumeGroups {
    std::vector<int> all; // Все объемы
    std::vector<int> heart; // Сердце
    std::vector<int> tissue; // Тело
};

/* ФУНКЦИИ ДЛЯ ПОСТРОЕНИЯ КОТА */

/* УШИ КОТА */
// baseCenter — центр нижней части уха, baseR — размеры нижнего эллипсоида, tipCenter — центр верхней части уха
// tipR — размеры верхнего эллипсоида, baseAng — поворот нижней части, tipAng — поворот верхней части
static int addEarSoft(const Vec3 &baseCenter, const Vec3 &baseR, const Vec3 &tipCenter, const Vec3 &tipR, const Vec3 &baseAng, const Vec3 &tipAng) {
    int e1 = addEllipsoid(baseCenter, baseR, baseAng);
    int e2 = addEllipsoid(tipCenter, tipR, tipAng);
    return fuseList({e1, e2});
}

/* ПЕРЕДНЯЯ ЛАПА КОТА */
static int buildFrontPaw(double s, double ys) {
    std::vector<int> parts; // Список частей лапки котика

    // Капсула, которая как "стержень" для лапы
    parts.push_back(addCapsule(s * Vec3{-0.120, ys * 0.052, 0.102}, s * Vec3{-0.210, ys * 0.050, 0.040}, s * 0.023));

    // Нижняя подушка лапы, к которой потом пальцы будут крепться
    parts.push_back(addEllipsoid(s * Vec3{-0.245, ys * 0.050, 0.018}, s * Vec3{0.052, 0.030, 0.018}, {0.0, 0.0, 0.0}));

    // Верхний мягкий объем лапы
    parts.push_back(addEllipsoid(s * Vec3{-0.228, ys * 0.050, 0.036}, s * Vec3{0.040, 0.028, 0.017}, {0.0, 0.0, 0.0}));

    // Пальчики передней лапы
    for(int i = 0; i < 4; ++i) {
        double toe = (-0.012 + 0.008 * i);
        parts.push_back(addEllipsoid(s * Vec3{-0.280 + 0.008 * std::abs(i - 1.5), ys * 0.050 + toe, 0.012}, s * Vec3{0.014, 0.010, 0.010}, {0.0, 0.0, 0.0}));
    }

    // Верхняя масса лапы
    parts.push_back(addEllipsoid(s * Vec3{-0.155, ys * 0.052, 0.080}, s * Vec3{0.050, 0.037, 0.038}, {0.0, 0.0, 0.0}));
    return fuseList(parts);
}

/* ЗАДНЯЯ ЛАПА КОТА*/
static int buildRearPaw(double s, double ys) {
    std::vector<int> parts;

    // Капсула, которая задает основную вытянутую форму задней лапы
    parts.push_back(addCapsule(s * Vec3{0.160, ys * 0.050, 0.090}, s * Vec3{0.232, ys * 0.048, 0.038}, s * 0.022));

    // Нижняя подушка задней лапы
    parts.push_back(addEllipsoid(s * Vec3{0.265, ys * 0.048, 0.018}, s * Vec3{0.050, 0.029, 0.017}, {0.0, 0.0, 0.0}));

    // Верхний мягкий объем лапы над подушкой
    parts.push_back(addEllipsoid(s * Vec3{0.248, ys * 0.048, 0.034}, s * Vec3{0.036, 0.026, 0.015}, {0.0, 0.0, 0.0}));

    // Пальчики задней лапы
    for(int i = 0; i < 4; ++i) {
        double toe = (-0.012 + 0.008 * i);
        parts.push_back(addEllipsoid(s * Vec3{0.295 - 0.008 * std::abs(i - 1.5), ys * 0.048 + toe, 0.012}, s * Vec3{0.013, 0.010, 0.009},{0.0, 0.0, 0.0}));
    }

    // Верхняя массивная часть задней лапы, ближе к телу
    parts.push_back(addEllipsoid(s * Vec3{0.205, ys * 0.050, 0.074}, s * Vec3{0.050, 0.040, 0.040},{0.0, 0.0, 0.0}));
    return fuseList(parts);
}

/* ГЛАЗА - маленькие эллипсоиды */
static std::vector<int> buildEyes(double s) {
    std::vector<int> parts;
    parts.push_back(addEllipsoid(s * Vec3{-0.286,  0.040, 0.135}, s * Vec3{0.016, 0.010, 0.012}, {deg(0), deg(-12), deg(-10)}));
    parts.push_back(addEllipsoid(s * Vec3{-0.286, -0.040, 0.135}, s * Vec3{0.016, 0.010, 0.012}, {deg(0), deg(-12), deg(10)}));
    return parts;
}

/* ХВОСТ - цепочка капсул */
// хвост становится тоньше к кончику
static int buildTail(double s) {
    std::vector<int> parts;
    std::vector<Vec3> p = {
        {0.250,  0.000, 0.076},
        {0.320,  0.000, 0.084},
        {0.390,  0.000, 0.094},
        {0.450,  0.000, 0.088},
        {0.472,  0.000, 0.074},
        {0.445,  0.000, 0.058},
        {0.392,  0.000, 0.046}
    };
    std::vector<double> rr = {0.030, 0.027, 0.024, 0.021, 0.018, 0.016};

    for(std::size_t i = 0; i < rr.size(); ++i) {
        parts.push_back(addCapsule(s * p[i], s * p[i + 1], s * rr[i]));
    }

    // Утолщение у основания хвоста
    parts.push_back(addEllipsoid(s * Vec3{0.240, 0.000, 0.076}, s * Vec3{0.074, 0.052, 0.046}, {deg(0), deg(4), deg(0)}));
    return fuseList(parts);
}

/* СБОРКА КОТА */
static int buildCat(double scale) {

    // Функция добаления эллипсиодов, кот почти весь состоит из эллипсоидов
    // E({центр}, {радиусы}, {углы}) - так проще задавать в будущем
    auto E = [&](const Vec3 &c, const Vec3 &r, const Vec3 &ang) {
        return addEllipsoid(scale * c, scale * r, {deg(ang.x), deg(ang.y), deg(ang.z)});
    };

    /* ТУЛОВИЩЕ */
    int torso = fuseList({
        E({ 0.10, 0.00, 0.120}, {0.215, 0.112, 0.092}, {0, 0, 2}), // Основной корпус
        E({-0.03, 0.00, 0.116}, {0.135, 0.100, 0.084}, {0, 0, -2}), // Грудь
        E({ 0.22, 0.00, 0.112}, {0.135, 0.105, 0.084}, {0, 0, 2}), // Таз / задняя часть
        E({ 0.11, 0.00, 0.170}, {0.178, 0.085, 0.046}, {0, 0, 1}), // Верх спины
        E({ 0.09, 0.00, 0.050}, {0.220, 0.092, 0.035}, {0, 0,  0}), // Низ живота
        E({-0.04, 0.060, 0.108}, {0.080, 0.048, 0.060}, {0, 0, 0}), // Плечи
        E({-0.04,-0.060, 0.108}, {0.080, 0.048, 0.060}, {0, 0, 0}), // Плечи
        E({ 0.20, 0.065, 0.104}, {0.085, 0.055, 0.064}, {0, 0, 0}), // Бедра
        E({ 0.20,-0.065, 0.104}, {0.085, 0.055, 0.064}, {0, 0, 0}) // Бедра
    });

    /* ГОЛОВА И МОРДОЧКА */
    int head = fuseList({
        E({-0.145, 0.00, 0.123}, {0.075, 0.072, 0.070}, {0, -2, -4}), // Шея
        E({-0.235, 0.00, 0.142}, {0.090, 0.085, 0.066}, {0, -3, -5}), // Череп
        E({-0.215, 0.00, 0.160}, {0.050, 0.052, 0.032}, {0, -3, -4}), // Верх головы / затылок
        E({-0.300, 0.00, 0.108}, {0.050, 0.062, 0.038}, {0, -4, -4}), // Мордочка
        E({-0.333, 0.00, 0.103}, {0.020, 0.026, 0.017}, {0, -5, -4}), // Передняя часть морды
        E({-0.320, 0.00, 0.093}, {0.018, 0.024, 0.011}, {0, -4, -4}), // Верхняя губа
        E({-0.312, 0.00, 0.079}, {0.030, 0.038, 0.015}, {0, -2, -3}), // Подбородок
        E({-0.314, 0.024, 0.099}, {0.026, 0.021, 0.014}, {0, -2, -5}), // Щёки
        E({-0.314,-0.024, 0.099}, {0.026, 0.021, 0.014}, {0, -2,  5}), // Щёки
        E({-0.342, 0.00, 0.100}, {0.010, 0.012, 0.006}, {0, -6, -4}) // Носик
    });

    /* УШИ - ЛЕВОЕ И ПРАВОЕ */
    int earL = addEarSoft(scale * Vec3{-0.232,  0.047, 0.152}, scale * Vec3{0.024, 0.016, 0.024}, scale * Vec3{-0.242,  0.060, 0.198}, scale * Vec3{0.020, 0.010, 0.040}, {deg( 8), deg(-8), deg(-18)}, {deg(12), deg(-8), deg(-18)});
    int earR = addEarSoft(scale * Vec3{-0.232, -0.047, 0.152}, scale * Vec3{0.024, 0.016, 0.024}, scale * Vec3{-0.242, -0.060, 0.198}, scale * Vec3{0.020, 0.010, 0.040}, {deg(-8), deg(-8), deg(18)}, {deg(-12), deg(-8), deg(18)});

    /* ПЕРЕДНИЕ ЛАПЫ */
    int frontL = buildFrontPaw(scale, 1.0);
    int frontR = buildFrontPaw(scale, -1.0);
    int frontBlock = fuseList({frontL, frontR, E({-0.145, 0.00, 0.072}, {0.070, 0.060, 0.030}, {0, 0, -1})});

    /* ЗАДНИЕ ЛАПЫ */
    int rearL = buildRearPaw(scale, +1.0);
    int rearR = buildRearPaw(scale, -1.0);

    /* ХВОСТ И ГЛАЗА */
    std::vector<int> eyes = buildEyes(scale);
    int tail = buildTail(scale);

    /* СБОРКА В ОДИН ОБЪЕМ */
    std::vector<int> catParts = {torso, head, earL, earR, frontBlock, rearL, rearR, tail};
    catParts.insert(catParts.end(), eyes.begin(), eyes.end());
    int cat = fuseList(catParts);

    gmsh::model::occ::removeAllDuplicates();
    return cat;
}

/* ВНУТРЕННИЕ ЗОНЫ ДЛЯ ТЕПЛОВОЙ ЗАДАЧИ */
static ThermalRegions makeThermalRegions(double scale) {
    ThermalRegions tr;

    // Сердце задаем как небольшой эллипсоид в грудной части.
    tr.heart = {
        scale * Vec3{-0.045, 0.000, 0.108},
        scale * Vec3{ 0.040, 0.034, 0.032}
    };

    // Сосуды к лапам добавляются в solve_cat_heat.py как отдельная расчетная
    // область по центрам ячеек. Явные тонкие капсулы здесь делают сетку хрупкой, кот не мешится

    return tr;
}

// Функция для встраивания сердца внутрь модели кота как отдельного объёма
static void embedThermalRegions(int cat, const ThermalRegions &tr) {
    std::vector<DimTag> tools;
    int heart = addEllipsoid(tr.heart.c, tr.heart.r, {0.0, 0.0, 0.0});
    tools.push_back({3, heart});
    std::vector<DimTag> out;
    std::vector<std::vector<DimTag>> outMap;
    gmsh::model::occ::fragment({{3, cat}}, tools, out, outMap, -1, true, true);
    gmsh::model::occ::removeAllDuplicates();
}

// Функция проверки принадлежности точки p области внутри сердца
static bool pointInHeart(const Vec3 &p, const ThermalRegions &tr) {
    Vec3 q = p - tr.heart.c;
    double x = q.x / tr.heart.r.x;
    double y = q.y / tr.heart.r.y;
    double z = q.z / tr.heart.r.z;
    return x * x + y * y + z * z <= 1.08;
}

// Функция делает именованные группы в сетке, чтобы потом при прописвыании нагрева было понятнее 
static void addPhysicalGroupIfNotEmpty(int dim, const std::vector<int> &tags, const std::string &name) {
    if(tags.empty()) {
        return;
    }
    int pg = gmsh::model::addPhysicalGroup(dim, tags);
    gmsh::model::setPhysicalName(dim, pg, name);
}

// Разделяем все объёмы модели на две группы: сердце и ткань
static VolumeGroups classifyVolumeGroups(const ThermalRegions &tr) {
    VolumeGroups g;
    std::vector<DimTag> vols;
    gmsh::model::getEntities(vols, 3);

    for(const auto &v : vols) {
        int tag = v.second;
        g.all.push_back(tag);

        double cx = 0.0, cy = 0.0, cz = 0.0;
        gmsh::model::occ::getCenterOfMass(3, tag, cx, cy, cz);
        Vec3 c{cx, cy, cz};

        if(pointInHeart(c, tr)) {
            g.heart.push_back(tag);
        } else {
            g.tissue.push_back(tag);
        }
    }

    return g;
}

static void createPhysicalGroups(const VolumeGroups &vg) {
    std::cerr << "Volume groups: all=" << vg.all.size() << ", tissue=" << vg.tissue.size() << ", heart=" << vg.heart.size() << "\n";
    addPhysicalGroupIfNotEmpty(3, vg.tissue, "tissue");
    addPhysicalGroupIfNotEmpty(3, vg.heart, "heart");
}

int main(int argc, char **argv) {
    gmsh::initialize();
    gmsh::model::add("heat_cat");

    // Задаем параметры кота тут
    double scale = 0.62;
    double lcMin = 0.007;
    double lcMax = 0.024;
    std::set<std::string> args(argv, argv + argc);

    configureMesh(lcMin, lcMax);
    int cat = buildCat(scale);
    ThermalRegions thermalRegions = makeThermalRegions(scale);
    embedThermalRegions(cat, thermalRegions);

    gmsh::model::occ::synchronize();

    VolumeGroups volumeGroups = classifyVolumeGroups(thermalRegions);
    createPhysicalGroups(volumeGroups);

    gmsh::model::mesh::generate(2);
    gmsh::model::mesh::generate(3);
    gmsh::model::mesh::optimize("Netgen");

    gmsh::write("cat_heat_model.msh");

    if(!args.count("-nopopup")) gmsh::fltk::run();

    gmsh::finalize();
    return 0;
}