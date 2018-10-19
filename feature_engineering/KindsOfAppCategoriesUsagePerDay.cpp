#include <iomanip>
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <set>
#include <unordered_map>
#include "csv.h"

typedef struct {
    char device_id[64];
    char package[64];
    unsigned long cls_label;
    char start[32];
    char close[32];
} AppCatRow;

typedef struct {
    char device_id[64];
    char category[64];
    double total;
    long row_id;
} KindsOfAppCatUsagePerDay;

typedef struct {
    int class_label;
    int subclass_label;
} AppCat;


static std::vector<KindsOfAppCatUsagePerDay> Xtrain_TotalAppCatKinds;
static std::vector<KindsOfAppCatUsagePerDay> Xtest_TotalAppCatKinds;

static std::unordered_map<std::string, AppCat> App2Category;
static unsigned long UnknowAppLabel = 0;

////////////////////////////////////////////////////////////////////////////////
static std::unordered_map<std::string, int> deviceid_train;
static std::unordered_map<std::string, int> deviceid_test;

static void read_train_devids(int start_idx = 0)
{
    const char* file = "../../Data/Demo/deviceid_train.tsv";
    io::CSVReader<3, io::trim_chars<' '>, io::no_quote_escape<'\t'> > in(file);

    char* id, *p, *q;
    int idx = start_idx;
    while (in.read_row(id, p, q)) {
        deviceid_train.insert(std::make_pair(id, idx));
        idx++;
    }
}

static void read_test_devids(int start_idx = 0)
{
    const char* file = "../../Data/Demo/deviceid_test.tsv";
    io::CSVReader<1, io::trim_chars<' '>, io::no_quote_escape<'\t'> > in(file);

    char* id;
    int idx = start_idx;
    while (in.read_row(id)) {
        deviceid_test.insert(std::make_pair(id, idx));
        idx++;
    }
}


static void read_app_categories()
{
    const char* file = "../../Data/Demo/package_label_encoded.tsv";
    io::CSVReader<3, io::trim_chars<' '>, io::no_quote_escape<'\t'> > in(file);

    char* d;
    int c, sc;
    UnknowAppLabel = 0;
    while (in.read_row(d, c, sc)) {
        AppCat cat;
        cat.class_label = c;
        cat.subclass_label = sc;
        App2Category.insert(std::make_pair(d, cat));
        if (c > UnknowAppLabel) {
            UnknowAppLabel = c;
        }
    }

    UnknowAppLabel += 1;
}

////////////////////////////////////////////////////////////////////////////////
static void process_device(const std::vector<AppCatRow>& rows)
{
    if (rows.empty()) {
        return;
    }

    std::unordered_map<int, std::set<int> > app_cat_kinds_per_day;

    for (auto& row : rows) {
        std::uint64_t start = std::stoll(row.start) / 1000;
        std::uint64_t close = std::stoll(row.close) / 1000;

        int start_day = start / (24 * 60 * 60);
        int close_day = close / (24 * 60 * 60);
        
        for (int day = start_day; day <= close_day; day++) {
            auto itor = app_cat_kinds_per_day.find(day);
            if (itor == app_cat_kinds_per_day.end()) {
                itor = app_cat_kinds_per_day.insert(std::make_pair(day, std::set<int>())).first;
            }
            itor->second.insert(row.cls_label);
        }
    }

    float total = 0;
    for (auto& app_cat_kinds : app_cat_kinds_per_day) {
        total += app_cat_kinds.second.size();
    }

    int total_day = app_cat_kinds_per_day.size();

    KindsOfAppCatUsagePerDay rec;
    strcpy(rec.device_id, rows[0].device_id);
    rec.total = total / total_day;

    auto itor = deviceid_train.find(rec.device_id);
    if (itor != deviceid_train.end()) {
        rec.row_id = itor->second;
        Xtrain_TotalAppCatKinds.push_back(rec);
    } else {
        itor = deviceid_test.find(rec.device_id);
        if (itor != deviceid_test.end()) {
            rec.row_id = itor->second;
            Xtest_TotalAppCatKinds.push_back(rec);
        }
    }
}

static void generate_kinds_of_app_cat_usage_per_day()
{
    AppCatRow row = { 0 };
    char* d, *p, *s, *c;
    char device_id[64] = { 0 };
    std::vector<AppCatRow> rows;

    unsigned long count = 0;

    io::CSVReader<4, io::trim_chars<' '>, io::no_quote_escape<'\t'> > in("../../Data/Demo/deviceid_package_start_close_sorted.tsv");
    while (in.read_row(d, p, s, c)) {
        if (_stricmp(device_id, d)) {
            process_device(rows);
            rows.clear();
            strcpy(device_id, d);
        }

        memset(&row, 0, sizeof(row));
        strcpy(row.device_id, d);
        strcpy(row.package, p);
        strcpy(row.start, s);
        strcpy(row.close, c);

        auto itor = App2Category.find(row.package);
        if (itor != App2Category.end()) {
            row.cls_label = itor->second.class_label;
        } else {
            row.cls_label = UnknowAppLabel;
        }

        rows.push_back(row);

        count++;
        if (!(count % 100000)) {
            std::cout << "processed rows: " << count << std::endl;
        }
    }

    if (!rows.empty()) {
        process_device(rows);
        rows.clear();
        device_id[0] = '\0';
    }
}

void extract_feature_kinds_of_app_cat_usage_per_day()
{
    read_train_devids();
    read_test_devids();
    read_app_categories();

    generate_kinds_of_app_cat_usage_per_day();

    std::sort(Xtrain_TotalAppCatKinds.begin(), Xtrain_TotalAppCatKinds.end(),
        [](const KindsOfAppCatUsagePerDay& lhs, const KindsOfAppCatUsagePerDay& rhs) {
        return lhs.row_id < rhs.row_id;
    });

    std::ofstream output;
    output.open("Xtrain_KindsOfAppCatUsagePerDay.tsv");
    if (output.is_open()) {
        for (auto& elem : Xtrain_TotalAppCatKinds) {
            output << elem.device_id << "\t" << std::fixed << std::setprecision(6) << elem.total << std::endl;
        }
    }
    output.close();

    std::sort(Xtest_TotalAppCatKinds.begin(), Xtest_TotalAppCatKinds.end(),
        [](const KindsOfAppCatUsagePerDay& lhs, const KindsOfAppCatUsagePerDay& rhs) {
        return lhs.row_id < rhs.row_id;
    });

    output.open("Xtest_KindsOfAppCatUsagePerDay.tsv");
    if (output.is_open()) {
        for (auto& elem : Xtest_TotalAppCatKinds) {
            output << elem.device_id << "\t" << std::fixed << std::setprecision(6) << elem.total << std::endl;
        }
    }
    output.close();
}