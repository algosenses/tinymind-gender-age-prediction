#include <fstream>
#include <vector>
#include <unordered_map>
#include <algorithm>
#include "csv.h"

typedef struct _Row {
    char device_id[64];
    char package[64];
    char start[32];
    char close[32];
} Row;

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

void generate_sorted_apps_start_close()
{
    Row row = { 0 };
    std::vector<Row> sortedRows;
    char* d, *p, *s, *c;

    io::CSVReader<4, io::trim_chars<' '>, io::no_quote_escape<'\t'> > in("../../Data/Demo/deviceid_package_start_close.tsv");
    while (in.read_row(d, p, s, c)) {
        strcpy(row.device_id, d);
        strcpy(row.package, p);
        strcpy(row.start, s);
        strcpy(row.close, c);
        sortedRows.push_back(row);
    }

    std::sort(sortedRows.begin(), sortedRows.end(),
        [](const Row& lhs, const Row& rhs) {
        int cmp = _stricmp(lhs.device_id, rhs.device_id);
        if (cmp < 0) {
            return true;
        } else if (cmp == 0) {
            if (_stricmp(lhs.package, rhs.package) < 0) {
                return true;
            } else {
                return false;
            }
        } else {
            return false;
        }
    });

    std::ofstream output;
    output.open("deviceid_package_start_close_sorted.tsv");
    if (output.is_open()) {
        for (auto& row : sortedRows) {
            output << row.device_id << "\t" << row.package << "\t" << row.start << "\t" << row.close << std::endl;
        }
    }
    output.close();
}

void flat_device_packages_list()
{
    typedef struct {
        char device_id[64];
        char package[64];
        long row_id;
    } DevPkgRow;

    std::vector<DevPkgRow> dev_pkgs;
    std::vector<DevPkgRow> train_dev_pkgs;
    std::vector<DevPkgRow> test_dev_pkgs;

    io::CSVReader<2, io::trim_chars<' '>, io::no_quote_escape<'\t'> > in("../../Data/Demo/deviceid_packages.tsv");
    char*d, *p;
    char line[(32 + 1) * 1000];
    while (in.read_row(d, p)) {
        DevPkgRow row;
        strcpy(row.device_id, d);
        line[0] = '\0';
        strcpy(line, p);
        char *pch = strtok(line, ",");
        while (pch != NULL) {
            strcpy(row.package, pch);
            dev_pkgs.push_back(row);
            pch = strtok(NULL, ",");
        }
    }

    read_train_devids();
    int start_idx = deviceid_train.size();
    read_test_devids();

    for (auto& row : dev_pkgs) {
        auto itor = deviceid_train.find(row.device_id);
        if (itor != deviceid_train.end()) {
            row.row_id = itor->second;
            train_dev_pkgs.push_back(row);
        } else {
            itor = deviceid_test.find(row.device_id);
            if (itor != deviceid_test.end()) {
                row.row_id = itor->second;
                test_dev_pkgs.push_back(row);
            } else {
                throw std::runtime_error("can not found device id!");
            }
        }
    }

    std::sort(train_dev_pkgs.begin(), train_dev_pkgs.end(),
        [](const DevPkgRow& lhs, const DevPkgRow& rhs) {
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

    std::sort(test_dev_pkgs.begin(), test_dev_pkgs.end(),
        [](const DevPkgRow& lhs, const DevPkgRow& rhs) {
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

    std::ofstream output;
    output.open("train_deviceid_package_flatten.tsv");
    if (output.is_open()) {
        for (auto& row : train_dev_pkgs) {
            output << row.device_id << "\t" << row.package << std::endl;
        }
    }
    output.close();

    output.open("test_deviceid_package_flatten.tsv");
    if (output.is_open()) {
        for (auto& row : test_dev_pkgs) {
            output << row.device_id << "\t" << row.package << std::endl;
        }
    }
    output.close();
}