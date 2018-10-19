
#include <iostream>
#include <iomanip>
#include <fstream>
#include <string>
#include <vector>
#include <unordered_map>
#include <unordered_set>

#include "csv.h"

typedef struct {
    char device_id[64];
    char category[64];
    int num;
    double usage;
} AppCatUsage;

static std::vector<AppCatUsage> DeviceAppCatUsage;
static std::unordered_map<std::string, int> deviceid_train;
static std::unordered_map<std::string, int> deviceid_test;
typedef struct {
    int row;
    int col;
    double data;
} SparseMatrixElement;

static std::vector<SparseMatrixElement> Xtrain_AppCatUsage;
static std::vector<SparseMatrixElement> Xtest_AppCatUsage;

static std::unordered_map<std::string, int> AppCatOneHotEncoder;

static const int MAX_APP_CATEGORIES = (288 + 1);

////////////////////////////////////////////////////////////////////////////////
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

static void write_sparse_matrix_file(const std::vector<SparseMatrixElement>& elements, const char* filename)
{
    std::ofstream output;
    output.open(filename);
    if (output.is_open()) {
        for (auto& elem : elements) {
            output << elem.row << "\t" << elem.col << "\t" << std::fixed << std::setprecision(6) << elem.data << std::endl;
        }
    }
    output.close();
}

static void process_category(const std::vector<AppCatUsage>& category)
{
    if (category.empty()) {
        return;
    }

    double usage = 0;
    for (auto& cat : category) {
        usage += cat.usage;
    }

    AppCatUsage catusage;
    strcpy(catusage.device_id, category[0].device_id);
    strcpy(catusage.category, category[0].category);
    catusage.num = category.size();
    catusage.usage = usage;
    DeviceAppCatUsage.push_back(catusage);
}

static void process_device(const std::vector<AppCatUsage>& rows)
{
    if (rows.empty()) {
        return;
    }

    std::vector<AppCatUsage> cat_rows(rows);

    char cat_id[64] = { 0 };
    std::vector<AppCatUsage> cats;

    std::sort(cat_rows.begin(), cat_rows.end(),
        [](const AppCatUsage& lhs, const AppCatUsage& rhs) {
        return (atoi(lhs.category) < atoi(rhs.category));
    });

    for (auto& row : cat_rows) {
        if (_stricmp(cat_id, row.category)) {
            process_category(cats);
            cats.clear();
            strcpy(cat_id, row.category);
        }

        cats.push_back(row);
    }

    if (!cats.empty()) {
        process_category(cats);
    }
}

int generate_appcat_usage()
{
    std::vector<AppCatUsage> rows;

    io::CSVReader<3, io::trim_chars<' '>, io::no_quote_escape<'\t'> > in("../../Data/Demo/app_usage_with_category_label.tsv");

    char*d, *c;
    double t;
    char device_id[64] = { 0 };
    while (in.read_row(d, c, t)) {
        if (_stricmp(device_id, d)) {
            process_device(rows);
            rows.clear();
            strcpy(device_id, d);
        }

        AppCatUsage usage;
        strcpy(usage.device_id, d);
        strcpy(usage.category, c);
        usage.usage = t;
        rows.push_back(usage);
    }

    if (!rows.empty()) {
        process_device(rows);
        rows.clear();
        device_id[0] = '\0';
    }

    std::ofstream output;
    output.open("appcat_usage.tsv");
    if (output.is_open()) {
        for (auto& row : DeviceAppCatUsage) {
            output << row.device_id << "\t" << row.category << "\t" << std::fixed << std::setprecision(6) << row.usage << "\t" << row.num << std::endl;
        }
    }
    output.close();

    return 0;
}

static void process_device_sparse_elements(std::vector<AppCatUsage>& rows)
{
    if (rows.empty()) {
        return;
    }

    const char* device_id = rows.front().device_id;
    auto itor = deviceid_train.find(device_id);
    if (itor != deviceid_train.end()) {
        int rowNum = itor->second;
        for (auto& item : rows) {
                SparseMatrixElement elem;
                elem.row = rowNum;
                elem.col = atoi(item.category);
                elem.data = item.usage;
                Xtrain_AppCatUsage.push_back(elem);
        }
    }

    itor = deviceid_test.find(device_id);
    if (itor != deviceid_test.end()) {
        int rowNum = itor->second;
        for (auto& item : rows) {
            SparseMatrixElement elem;
            elem.row = rowNum;
            elem.col = atoi(item.category);
            elem.data = item.usage;
            Xtest_AppCatUsage.push_back(elem);
        }
    }
}

void extract_feature_appcat_usage()
{
    read_train_devids();
    read_test_devids();

    io::CSVReader<4, io::trim_chars<' '>, io::no_quote_escape<'\t'> > in("../../Data/Demo/appcat_usage.tsv");

    char*d, *c;
    double t;
    int appnum;
    int row = 0;
    int col = 0;
    char device_id[64] = { 0 };
    std::vector<AppCatUsage> rows;
    AppCatUsage time;

    while (in.read_row(d, c, t, appnum)) {
        if (_stricmp(device_id, d)) {
            process_device_sparse_elements(rows);
            rows.clear();
            strcpy(device_id, d);
        }

        strcpy(time.device_id, d);
        strcpy(time.category, c);
        time.num = appnum;
        time.usage = t;
        rows.push_back(time);
    }

    if (!rows.empty()) {
        process_device_sparse_elements(rows);
        rows.clear();
        device_id[0] = '\0';
    }

    std::sort(Xtrain_AppCatUsage.begin(), Xtrain_AppCatUsage.end(),
        [](const SparseMatrixElement& lhs, const SparseMatrixElement& rhs) {
        return lhs.row < rhs.row;
    });

    write_sparse_matrix_file(Xtrain_AppCatUsage, "Xtrain_TimeOfAppCategoriesUsage.tsv");

    std::sort(Xtest_AppCatUsage.begin(), Xtest_AppCatUsage.end(),
        [](const SparseMatrixElement& lhs, const SparseMatrixElement& rhs) {
        return lhs.row < rhs.row;
    });

    write_sparse_matrix_file(Xtest_AppCatUsage, "Xtest_TimeOfAppCategoriesUsage.tsv");
}
