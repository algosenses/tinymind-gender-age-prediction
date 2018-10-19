// g++ -std=c++0x feat_extract.cpp -O2 -o feat_extract -lpthread
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
    char package[64];
    unsigned long count_val;
    double avg_val;
    long row_id;
} AppUsage;

typedef struct {
    char device_id[64];
    char package[64];
    char start[32];
    char close[32];
} Row;

static std::vector<AppUsage> DeviceAppUsage;
static std::unordered_map<std::string, int> deviceid_train;
static std::unordered_map<std::string, int> deviceid_test;
typedef struct {
    int row;
    int col;
    unsigned long count_val;
    double avg_val;
} SparseMatrixElement;

static std::vector<SparseMatrixElement> Xtrain_AppUsage;
static std::vector<SparseMatrixElement> Xtest_AppUsage;

static std::unordered_map<std::string, int> PackageOneHotEncoder;

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

////////////////////////////////////////////////////////////////////////////////
static void process_package(const std::vector<Row>& pkgs, unsigned long long total)
{
    if (pkgs.empty()) {
        return;
    }

    AppUsage app_use_time;

    double use_time = 0;
    for (auto& pkg : pkgs) {
        std::uint64_t start = std::stoll(pkg.start);
        std::uint64_t close = std::stoll(pkg.close);
        use_time += (close - start);
    }

    app_use_time.count_val = use_time / 1000;

    if (total > 1) {
        use_time /= total;
        if (use_time < 0.000001) {
            use_time = 0.000001;
        }
    } else {
        use_time /= 1000; // to seconds
    }

    strcpy(app_use_time.device_id, pkgs[0].device_id);
    strcpy(app_use_time.package, pkgs[0].package);
    app_use_time.avg_val = use_time;
    DeviceAppUsage.push_back(app_use_time);
}

static void process_device(const std::vector<Row>& rows)
{
    if (rows.empty()) {
        return;
    }

    std::uint64_t total_time = 0;

    for (auto& row : rows) {
        std::uint64_t start = std::stoll(row.start);
        std::uint64_t close = std::stoll(row.close);
        total_time += (close - start);
    }
    
    char pkg_id[64] = { 0 };
    std::vector<Row> pkgs;
    for (auto& row : rows) {
        if (_stricmp(pkg_id, row.package)) {
            process_package(pkgs, total_time);
            pkgs.clear();
            strcpy(pkg_id, row.package);
        }

        pkgs.push_back(row);
    }

    if (!pkgs.empty()) {
        process_package(pkgs, total_time);
    }
}

int generate_app_usage()
{
    Row row = { 0 };
    char* d, *p, *s, *c;
    char device_id[64] = { 0 };
    std::vector<Row> rows;

    unsigned long count = 0;

    io::CSVReader<4, io::trim_chars<' '>, io::no_quote_escape<'\t'> > in("../../Data/Demo/deviceid_package_start_close_sorted.tsv");
    while (in.read_row(d, p, s, c)) {
        if (_stricmp(device_id, d)) {
            process_device(rows);
            rows.clear();
            strcpy(device_id, d);
        }

        strcpy(row.device_id, d);
        strcpy(row.package, p);
        strcpy(row.start, s);
        strcpy(row.close, c);
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

    if (!DeviceAppUsage.empty()) {
        read_train_devids(0);
        int start_idx = deviceid_train.size();
        read_test_devids(start_idx);

        for (auto& row : DeviceAppUsage) {
            auto itor = deviceid_train.find(row.device_id);
            if (itor != deviceid_train.end()) {
                row.row_id = itor->second;
            } else {
                itor = deviceid_test.find(row.device_id);
                if (itor != deviceid_test.end()) {
                    row.row_id = itor->second;
                } else {
                    throw std::runtime_error("can not found device id!");
                }
            }
        }

        std::sort(DeviceAppUsage.begin(), DeviceAppUsage.end(), 
            [](const AppUsage& lhs, const AppUsage& rhs) {
            if (lhs.row_id < rhs.row_id) {
                return true;
            } else if (lhs.row_id == rhs.row_id) {
                if (_stricmp(lhs.package, rhs.package) < 0) {
                    return true;
                } else {
                    return false;
                }
            } else {
                return false;
            }
        });

        // merge same rows
        std::vector<AppUsage> dev_app_usage_time;
        for (auto& row : DeviceAppUsage) {
            if (!dev_app_usage_time.empty()) {
                auto& last = dev_app_usage_time.back();
                if (!_stricmp(row.device_id, last.device_id) &&
                    !_stricmp(row.package, last.package)) {
                    last.avg_val += row.avg_val;
                    last.count_val += row.count_val;
                } else {
                    dev_app_usage_time.push_back(row);
                }
            } else {
                dev_app_usage_time.push_back(row);
            }
        }

        std::ofstream output;
        output.open("deviceid_package_usage.tsv");
        if (output.is_open()) {
//            output << "device_id\tpackage\tusage_time\n";
            for (auto& row : dev_app_usage_time) {
                output << row.device_id << "\t" << row.package << "\t" << row.count_val << "\t" << std::fixed << std::setprecision(6) << row.avg_val << std::endl;
            }
        }
    }

    return 0;
}

////////////////////////////////////////////////////////////////////////////////
static void read_packages_onehot()
{
    std::cout << "read packages one-hot encoding..." << std::endl;
    std::unordered_set<std::string> pkg_ids;
    io::CSVReader<4, io::trim_chars<' '>, io::no_quote_escape<'\t'> > in("../../Data/Demo/deviceid_package_usage.tsv");

    char*d, *p, *a, *c;
    while (in.read_row(d, p, c, a)) {
        pkg_ids.insert(p);
    }

    int onehot = 0;
    for (auto& id : pkg_ids) {
        PackageOneHotEncoder.insert(std::make_pair(id, onehot));
        onehot++;
    }
}

static void process_device_sparse_elements(std::vector<AppUsage>& rows)
{
    if (rows.empty()) {
        return;
    }

    const char* device_id = rows.front().device_id;
    auto itor = deviceid_train.find(device_id);
    if (itor != deviceid_train.end()) {
        int rowNum = itor->second;
        for (auto& item : rows) {
            char* pkg_id = item.package;
            auto it = PackageOneHotEncoder.find(pkg_id);
            if (it != PackageOneHotEncoder.end()) {
                int colNum = it->second;
                SparseMatrixElement elem;
                elem.row = rowNum;
                elem.col = colNum;
                elem.count_val = item.count_val;
                elem.avg_val = item.avg_val;
                Xtrain_AppUsage.push_back(elem);
            }
        }
    }

    itor = deviceid_test.find(device_id);
    if (itor != deviceid_test.end()) {
        int rowNum = itor->second;
        for (auto& item : rows) {
            char* pkg_id = item.package;
            auto it = PackageOneHotEncoder.find(pkg_id);
            if (it != PackageOneHotEncoder.end()) {
                int colNum = it->second;
                SparseMatrixElement elem;
                elem.row = rowNum;
                elem.col = colNum;
                elem.avg_val = item.avg_val;
                elem.count_val = item.count_val;
                Xtest_AppUsage.push_back(elem);
            }
        }
    }
}

static void write_sparse_matrix_file(const std::vector<SparseMatrixElement>& elements, const char* filename)
{
    std::ofstream output;
    output.open(filename);
    if (output.is_open()) {
        for (auto& elem : elements) {
            output << elem.row << "\t" << elem.col << "\t" << elem.count_val << "\t" << std::fixed << std::setprecision(6) << elem.avg_val << std::endl;
        }
    }
    output.close();
}

void extract_feature_app_usage()
{
    read_train_devids();
    read_test_devids();

    read_packages_onehot();

    io::CSVReader<4, io::trim_chars<' '>, io::no_quote_escape<'\t'> > in("../../Data/Demo/deviceid_package_usage.tsv");


    char*d, *p;
    unsigned long c;
    double a;
    int row = 0;
    int col = 0;
    char device_id[64] = { 0 };
    std::vector<AppUsage> rows;
    AppUsage time;

    while (in.read_row(d, p, c, a)) {
        if (_stricmp(device_id, d)) {
            process_device_sparse_elements(rows);
            rows.clear();
            strcpy(device_id, d);
        }

        strcpy(time.device_id, d);
        strcpy(time.package, p);
        time.count_val = c;
        time.avg_val = a;
        rows.push_back(time);
    }

    if (!rows.empty()) {
        process_device_sparse_elements(rows);
        rows.clear();
        device_id[0] = '\0';
    }

    std::sort(Xtrain_AppUsage.begin(), Xtrain_AppUsage.end(),
        [](const SparseMatrixElement& lhs, const SparseMatrixElement& rhs) {
        return lhs.row < rhs.row;
    });

    write_sparse_matrix_file(Xtrain_AppUsage, "Xtrain_TimeOfAppsUsage.tsv");

    std::sort(Xtest_AppUsage.begin(), Xtest_AppUsage.end(),
        [](const SparseMatrixElement& lhs, const SparseMatrixElement& rhs) {
        return lhs.row < rhs.row;
    });

    write_sparse_matrix_file(Xtest_AppUsage, "Xtest_TimeOfAppsUsage.tsv");
}
