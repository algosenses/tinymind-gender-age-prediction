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
    char start[32];
    char close[32];
} Row;

typedef struct {
    char device_id[64];
    double total;
    long row_id;
} NumOfProcessUsagePerDay;

static std::vector<NumOfProcessUsagePerDay> Xtrain_NumOfProcess;
static std::vector<NumOfProcessUsagePerDay> Xtest_NumOfProcess;

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

////////////////////////////////////////////////////////////////////////////////
static void process_device(const std::vector<Row>& rows)
{
    if (rows.empty()) {
        return;
    }

    std::unordered_map<int, unsigned long> process_num_per_day;

    for (auto& row : rows) {
        std::uint64_t start = std::stoll(row.start) / 1000;
        std::uint64_t close = std::stoll(row.close) / 1000;

        int start_day = start / (24 * 60 * 60);
        int close_day = close / (24 * 60 * 60);

        for (int day = start_day; day <= close_day; day++) {
            auto itor = process_num_per_day.find(day);
            if (itor == process_num_per_day.end()) {
                itor = process_num_per_day.insert(std::make_pair(day, 0)).first;
            }
            itor->second++;
        }
    }

    float total = 0;
    for (auto& proc_num : process_num_per_day) {
        total += proc_num.second;
    }

    int total_day = process_num_per_day.size();

    NumOfProcessUsagePerDay rec;
    strcpy(rec.device_id, rows[0].device_id);
    rec.total = total / total_day;

    auto itor = deviceid_train.find(rec.device_id);
    if (itor != deviceid_train.end()) {
        rec.row_id = itor->second;
        Xtrain_NumOfProcess.push_back(rec);
    } else {
        itor = deviceid_test.find(rec.device_id);
        if (itor != deviceid_test.end()) {
            rec.row_id = itor->second;
            Xtest_NumOfProcess.push_back(rec);
        }
    }
}

static void generate_num_of_process_usage_per_day()
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
}

void extract_feature_num_of_process_usage_per_day()
{
    read_train_devids();
    read_test_devids();

    generate_num_of_process_usage_per_day();

    std::sort(Xtrain_NumOfProcess.begin(), Xtrain_NumOfProcess.end(),
        [](const NumOfProcessUsagePerDay& lhs, const NumOfProcessUsagePerDay& rhs) {
        return lhs.row_id < rhs.row_id;
    });

    std::ofstream output;
    output.open("Xtrain_NumOfProcessUsagePerDay.tsv");
    if (output.is_open()) {
        for (auto& elem : Xtrain_NumOfProcess) {
            output << elem.device_id << "\t" << std::fixed << std::setprecision(6) << elem.total << std::endl;
        }
    }
    output.close();

    std::sort(Xtest_NumOfProcess.begin(), Xtest_NumOfProcess.end(),
        [](const NumOfProcessUsagePerDay& lhs, const NumOfProcessUsagePerDay& rhs) {
        return lhs.row_id < rhs.row_id;
    });

    output.open("Xtest_NumOfProcessUsagePerDay.tsv");
    if (output.is_open()) {
        for (auto& elem : Xtest_NumOfProcess) {
            output << elem.device_id << "\t" << std::fixed << std::setprecision(6) << elem.total << std::endl;
        }
    }
    output.close();
}