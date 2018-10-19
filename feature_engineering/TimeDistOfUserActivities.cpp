#include <windows.h>
#include <time.h>
#include <iomanip>
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <unordered_map>
#include "csv.h"

typedef struct {
    char device_id[64];
    char package[64];
    char start[32];
    char close[32];
} Row_TimeDist;

const int seconds_per_hour = (60 * 60);
const int seconds_per_day = (24 * 60 * 60);
const int hours_per_day = 24;
typedef unsigned long SecondActivity_TimeDist[seconds_per_day];
typedef double HourActivity_TimeDist[24];

typedef struct {
    char device_id[64];
    HourActivity_TimeDist activity;
    int  most_active_hour;
    long row_id;
} UserActivity_TimeDist;

static std::unordered_map<std::string, int> deviceid_train;
static std::unordered_map<std::string, int> deviceid_test;

static std::vector<UserActivity_TimeDist> Xtrain_UserActivity;
static std::vector<UserActivity_TimeDist> Xtest_UserActivity;

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
static inline void update_activity(SecondActivity_TimeDist& activity, const std::uint64_t& start_ts, const std::uint64_t& close_ts)
{
    unsigned long start_sec = start_ts / 1000;    // ms to sec
    unsigned long close_sec = close_ts / 1000;
    if (close_sec < start_sec) {
        return;
    }

    for (unsigned long i = start_sec; i < close_sec; i++) {
        activity[i % seconds_per_day] += 1;
    }
}

static void process_device(const std::vector<Row_TimeDist>& rows)
{
    if (rows.empty()) {
        return;
    }

    std::uint64_t total_time = 0;
    SecondActivity_TimeDist sec_activity = { 0 };
    for (auto& row : rows) {
        std::uint64_t start = std::stoll(row.start);
        std::uint64_t close = std::stoll(row.close);
        if (start == close) {
            close += 1000; // +1s for default
        }
        total_time += (close - start);

        update_activity(sec_activity, start, close);
    }

    total_time /= 1000;
    if (total_time == 0) {
        return;
//        throw std::runtime_error("total time is ZERO!");
    }

    HourActivity_TimeDist activity = { 0 };
    for (int i = 0; i < hours_per_day; i++) {
        for (int j = 0; j < seconds_per_hour; j++) {
            activity[i] += sec_activity[i*seconds_per_hour + j];
        }
    }

    int most_active = 0;
    int most_active_hour = 0;
    for (int i = 0; i < hours_per_day; i++) {
        if (activity[i] > most_active) {
            most_active = activity[i];
            most_active_hour = i;
        }
    }

    for (int i = 0; i < hours_per_day; i++) {
        activity[i] /= total_time;
    }

    UserActivity_TimeDist act;
    strcpy(act.device_id, rows[0].device_id);
    memcpy(&act.activity, &activity, sizeof(HourActivity_TimeDist));
    act.most_active_hour = most_active_hour;

    auto itor = deviceid_train.find(act.device_id);
    if (itor != deviceid_train.end()) {
        act.row_id = itor->second;
        Xtrain_UserActivity.push_back(act);
    } else {
        itor = deviceid_test.find(act.device_id);
        if (itor != deviceid_test.end()) {
            act.row_id = itor->second;
            Xtest_UserActivity.push_back(act);
        }
    }
}

static void generate_user_activity()
{
    Row_TimeDist row = { 0 };
    char* d, *p, *s, *c;
    char device_id[64] = { 0 };
    std::vector<Row_TimeDist> rows;

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

static void write_sparse_matrix_file(const std::vector<UserActivity_TimeDist>& elements, const char* filename)
{
    std::ofstream output;
    output.open(filename);
    if (output.is_open()) {
        for (auto& elem : elements) {
            output << elem.device_id << ",";
            for (int i = 0; i < hours_per_day; i++) {
                double val = elem.activity[i];

                if (val > 0) {
                    output << std::fixed << std::setprecision(6) << val;
                } else {
                    output << "0";
                }
                output << ",";
            }
            output << elem.most_active_hour << std::endl;
        }
    }
    output.close();
}

void extract_feature_user_activity()
{
    read_train_devids();
    read_test_devids();

    generate_user_activity();

    std::sort(Xtrain_UserActivity.begin(), Xtrain_UserActivity.end(),
        [](const UserActivity_TimeDist& lhs, const UserActivity_TimeDist& rhs) {
        return lhs.row_id < rhs.row_id;
    });

    write_sparse_matrix_file(Xtrain_UserActivity, "Xtrain_TimeDistOfUserActivities.tsv");

    std::sort(Xtest_UserActivity.begin(), Xtest_UserActivity.end(),
        [](const UserActivity_TimeDist& lhs, const UserActivity_TimeDist& rhs) {
        return lhs.row_id < rhs.row_id;
    });

    write_sparse_matrix_file(Xtest_UserActivity, "Xtest_TimeDistOfUserActivities.tsv");
}
