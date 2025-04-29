// 标准库头文件
#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <functional>
#include <future>
#include <iomanip>
#include <iostream>
#include <map>
#include <mutex>
#include <numeric>
#include <queue>
#include <thread>
#include <tuple>
#include <vector>

// 系统相关头文件
#ifdef _WIN32
#include <windows.h>
#include <psapi.h>
#else
#include <sys/resource.h>
#endif

// 第三方库头文件
#include "../cpp_libraries/include/gmp.h"

// 编译示例
// Win: g++ -I../include -std=c++17 search.cpp -o search -L../lib -lgmp -o search.exe
// Mac: clang++ -std=c++17 search.cpp $(pkg-config --cflags --libs gmp) -o search

// 全局常量和变量
std::mutex cout_mutex;
std::condition_variable cv;
unsigned long current_threads = 0;
const unsigned long max_threads = std::max(1u, std::thread::hardware_concurrency()*3/4);
std::atomic<unsigned int> call_count{0};

// 全局配置数据
std::map<int, long double> a1_dict = {
    {11, 4.0}, {13, 3.82}, {17, 3.60}, {19, 3.53}
};
std::map<int, long double> a2_dict = {
    {11, 71.0}, {13, 74.0}, {17, 80.0}, {19, 84.0}
};

class ThreadPool {
public:
    explicit ThreadPool(size_t threads) : stop(false) {
        for(size_t i = 0; i < threads; ++i)
            workers.emplace_back([this] {
                while(true) {
                    std::function<void()> task;
                    {
                        std::unique_lock<std::mutex> lock(queue_mutex);
                        condition.wait(lock, [this] { 
                            return stop || !tasks.empty(); 
                        });
                        if(stop && tasks.empty()) return;
                        task = std::move(tasks.front());
                        tasks.pop();
                    }
                    task();
                }
            });
    }
    template<class F>
    auto enqueue(F&& f) -> std::future<decltype(f())> {
        using return_type = decltype(f());
        auto task = std::make_shared<std::packaged_task<return_type()>>(std::forward<F>(f));
        std::future<return_type> res = task->get_future();
        {
            std::unique_lock<std::mutex> lock(queue_mutex);
            if(stop) throw std::runtime_error("enqueue on stopped ThreadPool");
            tasks.emplace([task](){ (*task)(); });
        }
        condition.notify_one();
        return res;
    }
    ~ThreadPool() {
        {
            std::unique_lock<std::mutex> lock(queue_mutex);
            stop = true;
        }
        condition.notify_all();
        for(std::thread &worker: workers) worker.join();
    }
private:
    std::vector<std::thread> workers;
    std::queue<std::function<void()>> tasks;
    std::mutex queue_mutex;
    std::condition_variable condition;
    bool stop;
};

void wait_for_space() {
    // 如果当前并发线程数达到了最大值，等待
    std::unique_lock<std::mutex> lock(cout_mutex);
    cv.wait(lock, [] { return current_threads < max_threads; });
}

void signal_thread_complete() {
    // 线程完成后，通知其他线程可以启动
    std::lock_guard<std::mutex> lock(cout_mutex);
    current_threads--;
    cv.notify_all();
}

void write_program_start() {
    std::lock_guard<std::mutex> lock(cout_mutex);
    std::ofstream outfile("search_results.txt", std::ios::app);
    if (!outfile) {
        std::cerr << "Failed to open output file" << std::endl;
        return;
    }
    
    auto now = std::chrono::system_clock::now();
    auto now_time = std::chrono::system_clock::to_time_t(now);
    
    outfile << "\n========================================\n";
    outfile << "Program Execution Started\n";
    outfile << "Date and Time: " << std::ctime(&now_time);
    outfile << "System Information:\n";
    outfile << "Available CPU Threads: " << std::thread::hardware_concurrency() << "\n";
    outfile << "Maximum Concurrent Threads: " << max_threads << "\n";
    outfile << "========================================\n\n";
    outfile.close();
}

struct PerformanceMetrics {
    double wall_time;
    double cpu_time;
    size_t solutions_count;
    unsigned long long function_calls;
};

#ifdef _WIN32
double get_cpu_time_windows(const ULARGE_INTEGER& start_time) {
    FILETIME creation_time, exit_time, kernel_time, user_time;
    GetProcessTimes(GetCurrentProcess(), &creation_time, &exit_time, &kernel_time, &user_time);
    ULARGE_INTEGER end_time;
    end_time.LowPart = user_time.dwLowDateTime + kernel_time.dwLowDateTime;
    end_time.HighPart = user_time.dwHighDateTime + kernel_time.dwHighDateTime;
    return (end_time.QuadPart - start_time.QuadPart) / 10000000.0;
}
#else
double get_cpu_time_unix(const struct rusage& start_usage) {
    struct rusage end_usage;
    getrusage(RUSAGE_SELF, &end_usage);
    return ((end_usage.ru_utime.tv_sec - start_usage.ru_utime.tv_sec) +
            (end_usage.ru_utime.tv_usec - start_usage.ru_utime.tv_usec) * 1e-6) +
           ((end_usage.ru_stime.tv_sec - start_usage.ru_stime.tv_sec) +
            (end_usage.ru_stime.tv_usec - start_usage.ru_stime.tv_usec) * 1e-6);
}
#endif

void print_performance_statistics(const PerformanceMetrics& metrics, const std::vector<std::string>& solutions) {
    std::cout << "\nPerformance Statistics:" << std::endl;
    std::cout << "------------------------" << std::endl;
    std::cout << "Total solutions found: " << metrics.solutions_count << std::endl;
    for (const auto& solution : solutions) std::cout << solution << std::endl;
    std::cout << "Wall clock time: " << metrics.wall_time << " seconds (" 
              << std::fixed << std::setprecision(2) << (metrics.wall_time / 3600.0) << " hours)" << std::endl;
    std::cout << "Total CPU time: " << std::fixed << std::setprecision(2) 
              << (metrics.cpu_time / 3600.0) << " core*hours" << std::endl;
    std::cout << "Average CPU cores used: " << std::fixed << std::setprecision(2) 
              << (metrics.cpu_time / metrics.wall_time) << std::endl;
    std::cout << "Function calls: " << std::scientific << std::setprecision(2) 
              << static_cast<double>(metrics.function_calls) << std::endl;
}

void write_performance_statistics(const PerformanceMetrics& metrics, double cpu_time, size_t solutions_count, unsigned long long function_calls) {
    std::lock_guard<std::mutex> lock(cout_mutex);
    std::ofstream outfile("search_results.txt", std::ios::app);
    if (!outfile) {
        std::cerr << "Failed to write performance statistics" << std::endl;
        return;
    }
    
    outfile << "\nPerformance Statistics:" << std::endl;
    outfile << "------------------------" << std::endl;
    outfile << "Total solutions found: " << solutions_count << std::endl;
    outfile << "Wall clock time: " << metrics.wall_time << " seconds (" 
            << std::fixed << std::setprecision(2) << (metrics.wall_time / 3600.0) << " hours)" << std::endl;
    outfile << "Total CPU time: " << std::fixed << std::setprecision(2) 
            << (cpu_time / 3600.0) << " core*hours" << std::endl;
    outfile << "Average CPU cores used: " << std::fixed << std::setprecision(2) 
            << (cpu_time / metrics.wall_time) << std::endl;
    outfile << "Function calls: " << std::scientific << std::setprecision(2) 
            << static_cast<double>(function_calls) << std::endl;
    outfile << "========================================\n";
    outfile.close();
}

std::string mpz_to_string(const mpz_t n) {
    char* str = mpz_get_str(nullptr, 10, n);
    std::string result(str);
    free(str);
    return result;
}

bool is_prime(int n) {
    if (n < 2) return false;
    if (n == 2) return true;
    if (n % 2 == 0) return false;
    for (int i = 2; i * i <= n; ++i) if (n % i == 0) return false;
    return true;
}

std::vector<int> primes(int n) {
    if (n < 2) return {};
    std::vector<int> res;
    std::vector<bool> sieve(n + 1, true);
    for (int p = 2; p <= n; ++p) {
        if (sieve[p]) {
            res.push_back(p);
            for (int i = 2 * p; i <= n; i += p) sieve[i] = false;
        }
    }
    return res;
}

bool is_kth_power(const mpz_t n, unsigned int k) {
    call_count++;
    mpz_t root, power;
    mpz_init(root);
    mpz_init(power);
    mpz_root(root, n, k);
    mpz_pow_ui(power, root, k);
    bool result = (mpz_cmp(n, power) == 0);
    mpz_clear(root);
    mpz_clear(power);
    return result;
}

// check x1, r, y1, s, here l \nmid r * s
std::vector<std::string> check_x1_r_y1_s(unsigned long x1, int r, unsigned long y1, int s, int l, int t)
{
    std::vector<std::string> solutions;
    mpz_t xr, ys, zt, x1r, y1s, x1r_l, y1s_l, mpz_2, mpz_l, power, mpz_tmp;
    mpz_init(xr); mpz_init(ys); mpz_init(zt); mpz_init(x1r); mpz_init(y1s);
    mpz_init(x1r_l); mpz_init(y1s_l); mpz_init(mpz_2); mpz_init(mpz_l); mpz_init(power);
    mpz_init(mpz_tmp);
    mpz_set_ui(x1r, x1); mpz_set_ui(y1s, y1);
    mpz_pow_ui(x1r, x1r, r); mpz_pow_ui(y1s, y1s, s);
    mpz_set_ui(mpz_2, 2); mpz_set_ui(mpz_l, l);

    std::vector<std::pair<int, int>> possible_r2_s2;
    for (int r2 = 0; r2 <= 306 / r; ++r2) possible_r2_s2.push_back({r2, 0});
    for (int s2 = 0; s2 <= 306 / s; ++s2) possible_r2_s2.push_back({0, s2});
    std::vector<std::pair<int, int>> possible_rl_sl;
    for (int rl = 0; rl <= 37 / r; ++rl) possible_rl_sl.push_back({rl, 0});
    for (int sl = 0; sl <= 37 / s; ++sl) possible_rl_sl.push_back({0, sl});

    for (const auto &[rl, sl] : possible_rl_sl) {
        

        mpz_set(x1r_l, x1r); mpz_set(y1s_l, y1s);
        if (rl > 0) {
            mpz_pow_ui(power, mpz_l, rl * r);
            mpz_mul(x1r_l, x1r, power);
        }
        if (sl > 0) {
            mpz_pow_ui(power, mpz_l, sl * s);
            mpz_mul(y1s_l, y1s, power);
        }

        for (const auto &[r2, s2] : possible_r2_s2) {
            std::vector<std::pair<int, int>> possible_xl_yl = {{1, 1}};
            if (r == 4 && l == 11) {
                possible_xl_yl.push_back({3, 1}); possible_xl_yl.push_back({5, 1});
            } else if ((r == 4 && l == 13) || (r == 4 && l == 17) ||
                      (r == 5 && l == 11) || (r == 5 && l == 13) ||
                      (r == 6 && l == 11)) {
                possible_xl_yl.push_back({3, 1});
            }
            if (s == 4 && l == 11) {
                possible_xl_yl.push_back({1, 3}); possible_xl_yl.push_back({1, 5});
            } else if ((s == 4 && l == 13) || (s == 4 && l == 17) ||
                      (s == 5 && l == 11) || (s == 5 && l == 13) ||
                      (s == 6 && l == 11)) {
                possible_xl_yl.push_back({1, 3});
            }

            for (const auto &[xl, yl] : possible_xl_yl) {
                mpz_set(xr, x1r_l); mpz_set(ys, y1s_l);
                if (r2 > 0) {
                    mpz_pow_ui(power, mpz_2, r2 * r);
                    mpz_mul(xr, x1r_l, power);
                }
                if (s2 > 0) {
                    mpz_pow_ui(power, mpz_2, s2 * s);
                    mpz_mul(ys, y1s_l, power);
                }
                if (xl > 1) {
                    mpz_set_ui(mpz_tmp, xl);
                    mpz_pow_ui(power, mpz_tmp, l * r);
                    mpz_mul(xr, xr, power);
                }
                if (yl > 1) {
                    mpz_set_ui(mpz_tmp, yl);
                    mpz_pow_ui(power, mpz_tmp, l * s);
                    mpz_mul(ys, ys, power);
                }

                if (mpz_cmp(xr, ys) > 0) mpz_swap(xr, ys);
                if (mpz_cmp(xr, ys) == 0 || mpz_cmp_ui(xr, 0) == 0 || mpz_cmp_ui(xr, 1) == 0) continue;

                mpz_add(zt, xr, ys);
                if (is_kth_power(zt, t)) {
                    std::string solution = mpz_to_string(zt) + " + " + mpz_to_string(ys) + " = " + mpz_to_string(xr);
                    write_solution_to_file(solution, r, s, t);
                    solutions.push_back(solution);
                }

                mpz_sub(zt, ys, xr);
                if (is_kth_power(zt, t)) {
                    if (mpz_cmp(zt, xr) > 0) mpz_swap(xr, zt);
                    std::string solution = mpz_to_string(xr) + " + " + mpz_to_string(zt) + " = " + mpz_to_string(ys);
                    write_solution_to_file(solution, r, s, t);
                    solutions.push_back(solution);
                }
            }
        }
    }

    mpz_clear(xr); mpz_clear(ys); mpz_clear(zt); mpz_clear(x1r); mpz_clear(y1s);
    mpz_clear(x1r_l); mpz_clear(y1s_l); mpz_clear(mpz_2); mpz_clear(mpz_l); mpz_clear(power);mpz_clear(mpz_tmp);
    return solutions;
}

// check x1, r, here l \nmid r, y = 2^s, s <= 306
std::vector<std::string> check_x1_r(unsigned long x1, int r, int l, int t) {
    std::vector<std::string> solutions;
    mpz_t xr, ys, zt, x1r, x1r_l, mpz_2, mpz_l, power, mpz_tmp;
    mpz_init(xr); mpz_init(ys); mpz_init(zt); mpz_init(x1r); mpz_init(x1r_l);
    mpz_init(mpz_2); mpz_init(mpz_l); mpz_init(power);mpz_init(mpz_tmp);
    mpz_set_ui(x1r, x1); mpz_pow_ui(x1r, x1r, r);
    mpz_set_ui(mpz_2, 2); mpz_set_ui(mpz_l, l);

    std::vector<int> possible_xl_yl = {1};
            if (r == 4 && l == 11) {
                possible_xl_yl.push_back(3); possible_xl_yl.push_back(5);
            } else if ((r == 4 && l == 13) || (r == 4 && l == 17) ||
                     (r == 5 && l == 11) || (r == 5 && l == 13) ||
                     (r == 6 && l == 11)) {
                possible_xl_yl.push_back(3);
            }

    for (int rl = 0; rl <= 37 / r; ++rl) {
        mpz_set(x1r_l, x1r);
        if (rl > 0) {
            mpz_pow_ui(power, mpz_l, rl * r);
            mpz_mul(x1r_l, x1r, power);
        }
       
        for (int s2 = 70; s2 <= 306; ++s2) {

            for (auto xl : possible_xl_yl) {
                mpz_set(xr, x1r_l);
                mpz_set_ui(ys, 1);
                if (s2 > 0) mpz_pow_ui(ys, mpz_2, s2);
                if (xl > 1) {
                    mpz_set_ui(mpz_tmp, xl);
                    mpz_pow_ui(power, mpz_tmp, l * r);
                    mpz_mul(xr, xr, power);
                }

                if (mpz_cmp(xr, ys) > 0) mpz_swap(xr, ys);
                if (mpz_cmp(xr, ys) == 0 || mpz_cmp_ui(xr, 0) == 0 || mpz_cmp_ui(xr, 1) == 0) continue;

                mpz_add(zt, xr, ys);
                if (is_kth_power(zt, t)) {
                    std::string solution = mpz_to_string(zt) + " + " + mpz_to_string(ys) + " = " + mpz_to_string(xr);
                    write_solution_to_file(solution, r, s2, t);
                    solutions.push_back(solution);
                }

                mpz_sub(zt, ys, xr);
                if (is_kth_power(zt, t)) {
                    if (mpz_cmp(zt, xr) > 0) mpz_swap(xr, zt);
                    std::string solution = mpz_to_string(xr) + " + " + mpz_to_string(zt) + " = " + mpz_to_string(ys);
                    write_solution_to_file(solution, r, s2, t);
                    solutions.push_back(solution);
                }
            }
        }
    }

    mpz_clear(xr); mpz_clear(ys); mpz_clear(zt); mpz_clear(x1r); mpz_clear(x1r_l);
    mpz_clear(mpz_2); mpz_clear(mpz_l); mpz_clear(power);mpz_clear(mpz_tmp);
    return solutions;
}

void write_solution_to_file(const std::string& solution, int r, int s, int t = -1) {
    std::lock_guard<std::mutex> lock(cout_mutex);
    std::ofstream outfile("search_results.txt", std::ios::app);
    if (!outfile) {
        std::cerr << "Failed to open output file" << std::endl;
        return;
    }
    
    outfile << "\nFound solution for r=" << r << ", s=" << s;
    if (t != -1) {
        outfile << ", t=" << t;
    }
    outfile << "\nSolution: " << solution << "\n";
    outfile << "----------------------------------------\n";
    outfile.close();
    
    // 同时输出到控制台
    std::cout << "Found solution: " << solution << std::endl;
}

unsigned long gcd(unsigned long a, unsigned long b) {
    while (b != 0) {
        unsigned long t = b;
        b = a % b;
        a = t;
    }
    return a;
}

std::vector<std::string> search_for_r_s_and_t_ge_70(int r, int s, int l) {
    auto search_start = std::chrono::high_resolution_clock::now();
    
    #ifdef _WIN32
    FILETIME creation_time, exit_time, kernel_time, user_time;
    GetProcessTimes(GetCurrentProcess(), &creation_time, &exit_time, &kernel_time, &user_time);
    ULARGE_INTEGER search_start_cpu;
    search_start_cpu.LowPart = user_time.dwLowDateTime + kernel_time.dwLowDateTime;
    search_start_cpu.HighPart = user_time.dwHighDateTime + kernel_time.dwHighDateTime;
    #else
    struct rusage search_start_usage;
    getrusage(RUSAGE_SELF, &search_start_usage);
    #endif

    std::vector<std::string> solutions;
    std::mutex solutions_mutex;
    long double a1 = a1_dict[l], a2 = a2_dict[l];
    long double bound = a2 / (r + s - 2 * a1);
    unsigned long actual_bound = static_cast<unsigned long>(std::floor(std::exp(bound)));
    unsigned long total = (actual_bound + 1) / 2;
    std::atomic<unsigned long> progress{0};
    unsigned long progress_interval = std::min(std::max(static_cast<unsigned long>(200), total / 10000), total);
    const int num_threads = std::thread::hardware_concurrency();
    std::vector<std::future<void>> futures;
    ThreadPool thread_pool(num_threads);

    for (unsigned long x1 = 1; x1 <= actual_bound; x1 += 2) {
        futures.push_back(thread_pool.enqueue([&, x1]() {
            if ((++progress) % progress_interval == 0 || progress == total) {
                std::lock_guard<std::mutex> lock(cout_mutex);
                std::cout << "r=" << r << " s=" << s << " t>=70 Progress: " << std::fixed 
                         << std::setprecision(2) << (100.0 * progress / total) << "%" << std::endl;
            }
            if (x1%2==0 || x1%l==0) return;
            std::vector<std::string> result = check_x1_r(x1, r, l, s);
            if (!result.empty()) {
                std::lock_guard<std::mutex> lock(solutions_mutex);
                solutions.insert(solutions.end(), result.begin(), result.end());
            }
            if (r != s) {
                result = check_x1_r(x1, s, l, r);
                if (!result.empty()) {
                    std::lock_guard<std::mutex> lock(solutions_mutex);
                    solutions.insert(solutions.end(), result.begin(), result.end());
                }
            }
        }));
    }
    for (auto& future : futures) future.wait();
    auto search_end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> search_duration = search_end - search_start;
    
    double search_cpu_time;
    #ifdef _WIN32
    search_cpu_time = get_cpu_time_windows(search_start_cpu);
    #else
    search_cpu_time = get_cpu_time_unix(search_start_usage);
    #endif

    write_performance_statistics(metrics, search_cpu_time, solutions.size(), call_count);
    print_performance_statistics(metrics, solutions);

    return solutions;
}

std::vector<std::string> search_for_r_s_and_big_t(int r, int s, int t, int l) {
    auto search_start = std::chrono::high_resolution_clock::now();
    
    #ifdef _WIN32
    FILETIME creation_time, exit_time, kernel_time, user_time;
    GetProcessTimes(GetCurrentProcess(), &creation_time, &exit_time, &kernel_time, &user_time);
    ULARGE_INTEGER search_start_cpu;
    search_start_cpu.LowPart = user_time.dwLowDateTime + kernel_time.dwLowDateTime;
    search_start_cpu.HighPart = user_time.dwHighDateTime + kernel_time.dwHighDateTime;
    #else
    struct rusage search_start_usage;
    getrusage(RUSAGE_SELF, &search_start_usage);
    #endif

    std::vector<std::string> solutions;
    std::mutex solutions_mutex;
    long double a1 = a1_dict[l], a2 = a2_dict[l];
    unsigned long total = 0;
    const unsigned long z_max = static_cast<unsigned long>(std::floor(std::exp(a2 / (t - a1))));
    for (unsigned long z1 = 1; z1 <= z_max; z1 += 2) {
        unsigned long x1_bound = std::exp((a2 - (t - a1)*std::log(z1)) / (r + s - 2*a1));
        total += (x1_bound + 1) / 2;
        std::cout << "Total length of the progress bar: " << total << std::endl;
    }
    unsigned long progress = 0;
    unsigned long progress_interval = std::min(std::max(static_cast<unsigned long>(200), total / 10000), total);
    const int num_threads = std::thread::hardware_concurrency();
    std::vector<std::future<void>> futures;
    ThreadPool thread_pool(num_threads);
    
    for (unsigned long z1 = 1; z1 <= z_max; z1 += 2) {
        const unsigned long x1_bound = std::exp((a2 - (t - a1)*std::log(z1)) / (r + s - 2*a1));
        for (unsigned long x1 = 1; x1 <= x1_bound; x1 += 2) {
            futures.push_back(thread_pool.enqueue([&, z1, x1]() {
                if ((++progress) % progress_interval == 0 || progress == total) {
                    std::lock_guard<std::mutex> lock(cout_mutex);
                    std::cout << "r=" << r << " s=" << s << " t=" << t << " Progress: " << std::fixed 
                             << std::setprecision(2) << (100.0 * progress / total) << "%" << std::endl;
                }
                if (x1%2==0 || x1%l==0 || z1%l==0 || gcd(x1,z1)!=1) return;
                std::vector<std::string> result = check_x1_r_y1_s(x1, r, z1, t, l, s);
                if (!result.empty()) {
                    std::lock_guard<std::mutex> lock(solutions_mutex);
                    solutions.insert(solutions.end(), result.begin(), result.end());
                }
                if (r != s) {
                    result = check_x1_r_y1_s(x1, s, z1, t, l, r);
                    if (!result.empty()) {
                        std::lock_guard<std::mutex> lock(solutions_mutex);
                        solutions.insert(solutions.end(), result.begin(), result.end());
                    }
                }
            }));
        }
    }
    for (auto& future : futures) future.wait();
    auto search_end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> search_duration = search_end - search_start;
    
    double search_cpu_time;
    #ifdef _WIN32
    search_cpu_time = get_cpu_time_windows(search_start_cpu);
    #else
    search_cpu_time = get_cpu_time_unix(search_start_usage);
    #endif

    write_performance_statistics(metrics, search_cpu_time, solutions.size(), call_count);
    print_performance_statistics(metrics, solutions);

    return solutions;
}

std::vector<std::string> search_for_r_s_and_t(int r, int s, int t, int l) {
    auto search_start = std::chrono::high_resolution_clock::now();
    
    #ifdef _WIN32
    FILETIME creation_time, exit_time, kernel_time, user_time;
    GetProcessTimes(GetCurrentProcess(), &creation_time, &exit_time, &kernel_time, &user_time);
    ULARGE_INTEGER search_start_cpu;
    search_start_cpu.LowPart = user_time.dwLowDateTime + kernel_time.dwLowDateTime;
    search_start_cpu.HighPart = user_time.dwHighDateTime + kernel_time.dwHighDateTime;
    #else
    struct rusage search_start_usage;
    getrusage(RUSAGE_SELF, &search_start_usage);
    #endif

    std::vector<std::string> solutions;
    std::mutex solutions_mutex;
    const long double a1 = a1_dict[l], a2 = a2_dict[l];
    const unsigned long real_bound = static_cast<unsigned long>(std::floor(std::exp(2 * a2 / (r + s + t - 3 * a1))));
    unsigned long total = 0;
    for (unsigned long z1 = 1; z1 <= real_bound; z1 += 2) {
        total += (real_bound / z1 + 1) / 2;
        std::cout << "Total length of the progress bar: " << total << std::endl;
    }
    std::atomic<unsigned long> progress{0};
    unsigned long progress_interval = std::min(std::max(static_cast<unsigned long>(200), total / 10000), total);
    const int num_threads = std::thread::hardware_concurrency();
    ThreadPool thread_pool(num_threads);
    std::vector<std::future<void>> futures;

    for (unsigned long z1 = 1; z1 <= real_bound; z1 += 2) {
        const unsigned long x1_bound = real_bound / z1;
        for (unsigned long x1 = 1; x1 <= x1_bound; x1 += 2) {
            futures.push_back(thread_pool.enqueue([&, z1, x1]() {
                if ((++progress) % progress_interval == 0 || progress == total) {
                    std::lock_guard<std::mutex> lock(cout_mutex);
                    std::cout << "r=" << r << " s=" << s << " t=" << t << " Progress: " << std::fixed 
                             << std::setprecision(2) << (100.0 * progress / total) << "%" << std::endl;
                }
                if (x1%2==0 || x1%l==0 || z1%l==0 || gcd(x1,z1)!=1) return;
                std::vector<std::string> result = check_x1_r_y1_s(x1, r, z1, s, l, t);
                if (!result.empty()) {
                    std::lock_guard<std::mutex> lock(solutions_mutex);
                    solutions.insert(solutions.end(), result.begin(), result.end());
                }
                if (r != t) {
                    result = check_x1_r_y1_s(x1, s, z1, t, l, r);
                    if (!result.empty()) {
                        std::lock_guard<std::mutex> lock(solutions_mutex);
                        solutions.insert(solutions.end(), result.begin(), result.end());
                    }
                }
                if (s != t) {
                    result = check_x1_r_y1_s(x1, r, z1, t, l, s);
                    if (!result.empty()) {
                        std::lock_guard<std::mutex> lock(solutions_mutex);
                        solutions.insert(solutions.end(), result.begin(), result.end());
                    }
                }
            }));
        }
    }
    for (auto& future : futures) future.wait();
    auto search_end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> search_duration = search_end - search_start;
    
    double search_cpu_time;
    #ifdef _WIN32
    search_cpu_time = get_cpu_time_windows(search_start_cpu);
    #else
    search_cpu_time = get_cpu_time_unix(search_start_usage);
    #endif

    write_performance_statistics(metrics, search_cpu_time, solutions.size(), call_count);
    print_performance_statistics(metrics, solutions);

    return solutions;
}

std::vector<std::string> get_solutions_for_fixed_r_s_and_t(int r, int s, int t) {
    std::vector<std::string> solutions, result;
    // 交换 r, s, t 使得 4 <= r < s < t < 70
    if (r > s) std::swap(r, s);
    if (s > t) std::swap(s, t);
    if (r > s) std::swap(r, s);

    // 排除已证明的情况
    std::vector<std::pair<int, int>> pairs = {{2, 4}, {4, 2}, {2, 6}, {6, 2}, {3, 3}, {5, 5}};
    for (const auto& pair : pairs) {
        if (r % pair.first == 0 && s % pair.second == 0) return {};
        if (r % pair.first == 0 && t % pair.second == 0) return {};
        if (s % pair.first == 0 && t % pair.second == 0) return {};
    }

    std::vector<std::tuple<int, int, int>> tuples = {
        {2, 3, 7}, {2, 3, 8}, {2, 3, 9}, {2, 3, 10},
        {2, 7, 7}, {3, 4, 5}, {3, 7, 7}, {5, 7, 7}, {7, 7, 7}
    };
    
    std::array<int, 3> nums = {r, s, t};
    do {
        for (const auto& tuple : tuples) {
            if (nums[0] % std::get<0>(tuple) == 0 && 
                nums[1] % std::get<1>(tuple) == 0 && 
                nums[2] % std::get<2>(tuple) == 0) return {};
        }
    } while (std::next_permutation(nums.begin(), nums.end()));

    int l = -1;
    for (int l1 : {13, 17, 19, 11}) {
        if ((r * s * t) % l1 != 0) {
            l = l1;
            break;
        }
    }

    return (t >= r + s - 3) ? search_for_r_s_and_big_t(r, s, t, l) 
                            : search_for_r_s_and_t(r, s, t, l);
}

std::vector<std::string> get_solutions_for_fixed_r_s_and_t_ge_u0(int r, int s, int u0) {
    std::vector<std::string> solutions, result;
    std::vector<std::pair<int, int>> pairs = {{2, 4}, {4, 2}, {2, 6}, {6, 2}, {3, 3}, {5,5}};
    for (const auto& pair : pairs) {
        if (r % pair.first == 0 && s % pair.second == 0) return {};
    }

    int l = -1;
    for (int l1 : {13, 17, 19, 11}) {
        if ((r * s) % l1 != 0) {
            l = l1;
            break;
        }
    }

    result = search_for_r_s_and_t_ge_70(r, s, l);
    solutions.insert(solutions.end(), result.begin(), result.end());
    
    for (int t = u0; t < 70; ++t) {
        result = get_solutions_for_fixed_r_s_and_t(r, s, t);
        solutions.insert(solutions.end(), result.begin(), result.end());
    }
    return solutions;
}


int main() {
    auto wall_start = std::chrono::high_resolution_clock::now();
    
    #ifdef _WIN32
    FILETIME creation_time, exit_time, kernel_time, user_time;
    GetProcessTimes(GetCurrentProcess(), &creation_time, &exit_time, &kernel_time, &user_time);
    ULARGE_INTEGER start_cpu_time;
    start_cpu_time.LowPart = user_time.dwLowDateTime + kernel_time.dwLowDateTime;
    start_cpu_time.HighPart = user_time.dwHighDateTime + kernel_time.dwHighDateTime;
    #else
    struct rusage start_usage;
    getrusage(RUSAGE_SELF, &start_usage);
    #endif

    write_program_start();
    std::vector<std::string> solutions, result;
    
    // search code

    // example for (r,s,t) = (11,13,17)
    // result = get_solutions_for_fixed_r_s_and_t(11,13,17);
    /*
    Performance Statistics:
    ------------------------
    Total solutions found: 0
    Wall clock time: 0.10184 seconds (0.00 hours)
    Total CPU time: 0.00 core*hours
    Average CPU cores used: 7.01
    Function calls: 5.41e+05
    */
    

    // // search for 4 <= r <= s <= t (note that we have 4 <= r,s < 70) 
    
    // // the basic case: r + s >= 14
    // for (int r = 4; r < 70; ++r) {
    //     for (int s = r; s < 70 && r + s >= 14; ++s) {
    //         result = get_solutions_for_fixed_r_s_and_t_ge_u0(r, s, s);
    //         if (!result.empty()) solutions.insert(solutions.end(), result.begin(), result.end());
    //     }
    // }
    /*
    Performance Statistics:
    ------------------------
    Total solutions found: 0
    Wall clock time: 247.73 seconds (0.07 hours)
    Total CPU time: 0.50 core*hours
    Average CPU cores used: 7.32
    Function calls: 1.66e+09
    */

    // // the case of (r,s) = (4,9)
    // result = get_solutions_for_fixed_r_s_and_t_ge_u0(4, 9, 9);
    // if (!result.empty()) solutions.insert(solutions.end(), result.begin(), result.end());
    /*
    Performance Statistics:
    ------------------------
    Total solutions found: 0
    Wall clock time: 2688.62 seconds (0.75 hours)
    Total CPU time: 5.69 core*hours
    Average CPU cores used: 7.62
    Function calls: 3.44e+09
    */

    // the case of (r,s) = (5,8)
    // result = get_solutions_for_fixed_r_s_and_t_ge_u0(5, 8, 8);
    // if (!result.empty()) solutions.insert(solutions.end(), result.begin(), result.end());
    /*
    Performance Statistics:
    ------------------------
    Total solutions found: 0
    Wall clock time: 1994.38 seconds (0.55 hours)
    Total CPU time: 4.26 core*hours
    Average CPU cores used: 7.70
    Function calls: 2.73e+08
    */ 

    // the case of (r,s) = (6,7)
    // result = get_solutions_for_fixed_r_s_and_t_ge_u0(6, 7, 7);
    // if (!result.empty()) solutions.insert(solutions.end(), result.begin(), result.end());
    /* 
    Performance Statistics:
    ------------------------
    Total solutions found: 0
    Wall clock time: 1316.99 seconds (0.37 hours)
    Total CPU time: 2.77 core*hours
    Average CPU cores used: 7.56
    Function calls: 1.58e+08
    */ 

    // the case of (r,s) = (5,7,t), t >= 10
    //   for (int t = 69; t >= 10; --t){
    //     result = get_solutions_for_fixed_r_s_and_t(5, 7, t);
    //     if (!result.empty()) solutions.insert(solutions.end(), result.begin(), result.end());
    // }


    // result = get_solutions_for_fixed_r_s_and_t(5, 7, 9);
    // result = get_solutions_for_fixed_r_s_and_t(5, 7, 8);



    // left cases: (r,s) = (4,5), (4,7), (5,6)
    /* ??? */






    auto wall_end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> wall_duration = wall_end - wall_start;
    
    PerformanceMetrics metrics;
    metrics.wall_time = wall_duration.count();
    #ifdef _WIN32
    metrics.cpu_time = get_cpu_time_windows(start_cpu_time);
    #else
    metrics.cpu_time = get_cpu_time_unix(start_usage);
    #endif
    metrics.solutions_count = solutions.size();
    metrics.function_calls = call_count;

    write_performance_statistics(metrics, metrics.cpu_time, metrics.solutions_count, metrics.function_calls);
    print_performance_statistics(metrics, solutions);

    return 0;
}