#include "uno_constants.hpp"
#include "uno_engine.hpp"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

#include <vector>
#include <queue>
#include <thread>
#include <future>
#include <atomic>
#include <condition_variable>
#include <mutex>
#include <functional>
#include <memory>
#include <algorithm>
#include <string>
#include <cstdint>
#include <stdexcept>
#include <cstddef>
#include <random>

namespace py = pybind11;
using namespace uno;

// ============================================================================
// Lightweight Persistent Thread Pool for Sub-Millisecond Multi-Threaded Stepping
// ============================================================================
class SimpleThreadPool {
public:
    explicit SimpleThreadPool(size_t threads) : active_tasks_(0), stop_(false) {
        if (threads == 0) {
            threads = std::max(1u, std::thread::hardware_concurrency());
        }
        for (size_t i = 0; i < threads; ++i) {
            workers_.emplace_back([this]() {
                while (true) {
                    std::function<void()> task;
                    {
                        std::unique_lock<std::mutex> lock(queue_mutex_);
                        cv_.wait(lock, [this]() { return stop_ || !tasks_.empty(); });
                        if (stop_ && tasks_.empty()) return;
                        task = std::move(tasks_.front());
                        tasks_.pop();
                    }
                    task();
                    {
                        std::unique_lock<std::mutex> lock(done_mutex_);
                        --active_tasks_;
                        done_cv_.notify_all();
                    }
                }
            });
        }
    }

    size_t size() const { return workers_.size(); }

    void enqueue_batch(const std::vector<std::function<void()>>& tasks) {
        {
            std::unique_lock<std::mutex> lock(queue_mutex_);
            for (const auto& task : tasks) {
                tasks_.push(task);
            }
            active_tasks_ += tasks.size();
        }
        cv_.notify_all();
    }

    void wait_all() {
        std::unique_lock<std::mutex> lock(done_mutex_);
        done_cv_.wait(lock, [this]() {
            std::unique_lock<std::mutex> q_lock(queue_mutex_);
            return tasks_.empty() && active_tasks_ == 0;
        });
    }

    ~SimpleThreadPool() {
        {
            std::unique_lock<std::mutex> lock(queue_mutex_);
            stop_ = true;
        }
        cv_.notify_all();
        for (std::thread& worker : workers_) {
            if (worker.joinable()) {
                worker.join();
            }
        }
    }

private:
    std::vector<std::thread> workers_;
    std::queue<std::function<void()>> tasks_;
    std::mutex queue_mutex_;
    std::condition_variable cv_;
    std::mutex done_mutex_;
    std::condition_variable done_cv_;
    std::atomic<size_t> active_tasks_;
    bool stop_;
};

// ============================================================================
// Multi-Threaded Vectorized Environment
// ============================================================================
class VecUnoEnv {
public:
    int num_envs;
    uint64_t base_seed;
    bool auto_reset;
    std::vector<UnoEngine> engines;
    std::unique_ptr<SimpleThreadPool> pool;

    // Contiguous Host Buffers (Zero float64!)
    std::vector<float> card_features_buf;    // [B, 54, 7]
    std::vector<float> global_features_buf;  // [B, 10]
    std::vector<int32_t> history_tokens_buf; // [B, 48, 5]
    std::vector<uint8_t> legal_mask_buf;     // [B, 61]
    std::vector<float> opp_hand_counts_buf;  // [B, 54]
    std::vector<int32_t> current_player_buf; // [B]
    std::vector<int32_t> acting_player_buf;  // [B]
    std::vector<float> rewards_buf;          // [B]
    std::vector<uint8_t> dones_buf;          // [B]

    std::atomic<uint64_t> total_steps_{0};
    std::atomic<uint64_t> total_episodes_{0};

public:
    VecUnoEnv(int num_envs, uint64_t seed = 42, int num_threads = 0, bool auto_reset = true)
        : num_envs(num_envs), base_seed(seed), auto_reset(auto_reset) {

        if (num_threads <= 0) {
            num_threads = std::max(1u, std::thread::hardware_concurrency());
        }
        pool = std::make_unique<SimpleThreadPool>(num_threads);

        // Preallocate contiguous buffers
        card_features_buf.resize(num_envs * 54 * 7);
        global_features_buf.resize(num_envs * 10);
        history_tokens_buf.resize(num_envs * 48 * 5);
        legal_mask_buf.resize(num_envs * 61);
        opp_hand_counts_buf.resize(num_envs * 54);
        current_player_buf.resize(num_envs);
        acting_player_buf.resize(num_envs);
        rewards_buf.resize(num_envs);
        dones_buf.resize(num_envs);

        engines.reserve(num_envs);
        for (int i = 0; i < num_envs; ++i) {
            engines.emplace_back(seed + (uint64_t)i * 10007 + 1);
        }

        reset();
    }

    void resize(int new_num_envs) {
        if (new_num_envs == num_envs) return;
        num_envs = new_num_envs;
        card_features_buf.resize(num_envs * 54 * 7);
        global_features_buf.resize(num_envs * 10);
        history_tokens_buf.resize(num_envs * 48 * 5);
        legal_mask_buf.resize(num_envs * 61);
        opp_hand_counts_buf.resize(num_envs * 54);
        current_player_buf.resize(num_envs);
        acting_player_buf.resize(num_envs);
        rewards_buf.resize(num_envs);
        dones_buf.resize(num_envs);
        engines.resize(num_envs);
    }

    void _update_env_buffers(int env_idx) {
        const auto& engine = engines[env_idx];
        int8_t p = engine.current_player;

        engine.encode_card_features_actor(&card_features_buf[env_idx * 54 * 7], p);
        engine.encode_global_features(&global_features_buf[env_idx * 10], p);
        engine.encode_history_tokens(&history_tokens_buf[env_idx * 48 * 5], p);
        engine.get_action_mask(&legal_mask_buf[env_idx * 61], p);
        engine.encode_opp_hand_counts(&opp_hand_counts_buf[env_idx * 54], p);
        current_player_buf[env_idx] = static_cast<int32_t>(p);
    }

    py::dict reset() {
        size_t n_workers = pool->size();
        std::vector<std::function<void()>> tasks;
        tasks.reserve(n_workers);

        for (size_t t = 0; t < n_workers; ++t) {
            int start = static_cast<int>(t * num_envs / n_workers);
            int end = static_cast<int>((t + 1) * num_envs / n_workers);
            tasks.emplace_back([this, start, end]() {
                for (int i = start; i < end; ++i) {
                    engines[i].reset();
                    rewards_buf[i] = 0.0f;
                    dones_buf[i] = 0;
                    _update_env_buffers(i);
                }
            });
        }
        pool->enqueue_batch(tasks);
        pool->wait_all();

        return get_obs_dict();
    }

    py::tuple step(const py::array_t<int32_t, py::array::c_style | py::array::forcecast>& actions) {
        return _step_internal(actions, this->auto_reset);
    }

    py::tuple step_no_autoreset(const py::array_t<int32_t, py::array::c_style | py::array::forcecast>& actions) {
        return _step_internal(actions, false);
    }

    py::tuple _step_internal(const py::array_t<int32_t>& actions, bool do_autoreset) {
        py::buffer_info act_buf = actions.request();
        if (act_buf.size != num_envs) {
            throw std::runtime_error("Actions array size (" + std::to_string(act_buf.size) +
                                     ") must match num_envs (" + std::to_string(num_envs) + ")");
        }
        const int32_t* act_ptr = static_cast<const int32_t*>(act_buf.ptr);

        // Step multi-threaded without holding the Python GIL
        {
            py::gil_scoped_release release;

            size_t n_workers = pool->size();
            std::vector<std::function<void()>> tasks;
            tasks.reserve(n_workers);

            for (size_t t = 0; t < n_workers; ++t) {
                int start = static_cast<int>(t * num_envs / n_workers);
                int end = static_cast<int>((t + 1) * num_envs / n_workers);

                tasks.emplace_back([this, start, end, act_ptr, do_autoreset]() {
                    for (int i = start; i < end; ++i) {
                        auto& eng = engines[i];
                        acting_player_buf[i] = eng.current_player;
                        uint8_t a = static_cast<uint8_t>(act_ptr[i]);
                        // Execute action
                        float r = eng.step(a);
                        rewards_buf[i] = r;

                        bool done = eng.is_over;
                        dones_buf[i] = done ? 1 : 0;

                        if (done) {
                            total_episodes_.fetch_add(1, std::memory_order_relaxed);
                            if (do_autoreset) {
                                eng.reset();
                            }
                        }
                        _update_env_buffers(i);
                    }
                });
            }

            pool->enqueue_batch(tasks);
            pool->wait_all();
            total_steps_.fetch_add(num_envs, std::memory_order_relaxed);
        }

        py::dict obs = get_obs_dict();

        // Build 1D rewards and dones numpy arrays
        py::array_t<float> py_rewards(num_envs, rewards_buf.data());
        py::array_t<bool> py_dones(num_envs, reinterpret_cast<const bool*>(dones_buf.data()));
        py::dict info;

        return py::make_tuple(obs, py_rewards, py_dones, info);
    }

    // High speed in-engine random legal stepper for maximum benchmark throughput
    void step_random_legal() {
        size_t n_workers = pool->size();
        std::vector<std::function<void()>> tasks;
        tasks.reserve(n_workers);

        for (size_t t = 0; t < n_workers; ++t) {
            int start = static_cast<int>(t * num_envs / n_workers);
            int end = static_cast<int>((t + 1) * num_envs / n_workers);

            tasks.emplace_back([this, start, end]() {
                for (int i = start; i < end; ++i) {
                    auto& eng = engines[i];
                    uint8_t mask[61];
                    eng.get_action_mask(mask);
                    // Collect legal actions
                    uint8_t legal_ids[61];
                    int n_legal = 0;
                    for (int a = 0; a < 61; ++a) {
                        if (mask[a]) legal_ids[n_legal++] = a;
                    }
                    if (n_legal == 0) {
                        legal_ids[0] = 60; // draw fallback
                        n_legal = 1;
                    }
                    // Sample uniform legal action
                    int chosen_idx = eng.rng() % n_legal;
                    uint8_t act = legal_ids[chosen_idx];

                    float r = eng.step(act);
                    rewards_buf[i] = r;
                    bool done = eng.is_over;
                    dones_buf[i] = done ? 1 : 0;

                    if (done) {
                        total_episodes_.fetch_add(1, std::memory_order_relaxed);
                        if (auto_reset) {
                            eng.reset();
                        }
                    }
                    _update_env_buffers(i);
                }
            });
        }

        pool->enqueue_batch(tasks);
        pool->wait_all();
        total_steps_.fetch_add(num_envs, std::memory_order_relaxed);
    }

    py::dict get_obs_dict() {
        py::dict d;
        // Zero-copy array views pointing to host memory
        d["card_features_actor"] = py::array_t<float>(
            {num_envs, 54, 7},
            {54 * 7 * (ssize_t)sizeof(float), 7 * (ssize_t)sizeof(float), (ssize_t)sizeof(float)},
            card_features_buf.data(),
            py::cast(this)
        );

        d["global_features"] = py::array_t<float>(
            {num_envs, 10},
            {10 * (ssize_t)sizeof(float), (ssize_t)sizeof(float)},
            global_features_buf.data(),
            py::cast(this)
        );

        d["history_tokens"] = py::array_t<int32_t>(
            {num_envs, 48, 5},
            {48 * 5 * (ssize_t)sizeof(int32_t), 5 * (ssize_t)sizeof(int32_t), (ssize_t)sizeof(int32_t)},
            history_tokens_buf.data(),
            py::cast(this)
        );

        d["legal_mask"] = py::array_t<bool>(
            {num_envs, 61},
            {61 * (ssize_t)sizeof(uint8_t), (ssize_t)sizeof(uint8_t)},
            reinterpret_cast<const bool*>(legal_mask_buf.data()),
            py::cast(this)
        );

        d["opp_hand_counts"] = py::array_t<float>(
            {num_envs, 54},
            {54 * (ssize_t)sizeof(float), (ssize_t)sizeof(float)},
            opp_hand_counts_buf.data(),
            py::cast(this)
        );

        d["current_player"] = py::array_t<int32_t>(
            {num_envs},
            {(ssize_t)sizeof(int32_t)},
            current_player_buf.data(),
            py::cast(this)
        );

        d["acting_player"] = py::array_t<int32_t>(
            {num_envs},
            {(ssize_t)sizeof(int32_t)},
            acting_player_buf.data(),
            py::cast(this)
        );

        return d;
    }

    py::tuple load_worlds_and_step_root(
        const UnoEngine& root_engine,
        const py::array_t<int32_t, py::array::c_style | py::array::forcecast>& sampled_opp_hands,
        const py::array_t<int32_t, py::array::c_style | py::array::forcecast>& legal_actions,
        uint64_t seed = 42
    ) {
        py::buffer_info opp_buf = sampled_opp_hands.request();
        py::buffer_info act_buf = legal_actions.request();

        if (opp_buf.ndim != 2 || opp_buf.shape[1] != 54) {
            throw std::runtime_error("sampled_opp_hands must have shape [K, 54]");
        }
        int K = static_cast<int>(opp_buf.shape[0]);
        int num_a = static_cast<int>(act_buf.size);
        if (K <= 0 || num_a <= 0) {
            throw std::runtime_error("K and num_a must be positive");
        }

        int total_slots = num_a * K;
        if (total_slots != num_envs) {
            resize(total_slots);
        }

        const int32_t* opp_ptr = static_cast<const int32_t*>(opp_buf.ptr);
        const int32_t* act_ptr = static_cast<const int32_t*>(act_buf.ptr);

        int8_t root_p = root_engine.current_player;
        int8_t opp_p = 1 - root_p;

        uint8_t my_counts[54] = {0};
        for (uint8_t c : root_engine.hands[root_p]) {
            if (c < 54) my_counts[c]++;
        }
        const auto& full_counts = get_full_108_deck_counts();

        // Step 1: Pre-generate the K sampled worlds (opp_hand_k and draw_pile_k)
        struct WorldData {
            std::vector<uint8_t> opp_hand;
            std::vector<uint8_t> draw_pile;
        };
        std::vector<WorldData> worlds(K);

        for (int k = 0; k < K; ++k) {
            auto& w = worlds[k];
            w.opp_hand.reserve(16);
            w.draw_pile.reserve(80);

            for (int c = 0; c < 54; ++c) {
                int opp_cnt = opp_ptr[k * 54 + c];
                for (int i = 0; i < opp_cnt; ++i) {
                    w.opp_hand.push_back(static_cast<uint8_t>(c));
                }

                int rem = (int)full_counts[c] - (int)my_counts[c] - (int)root_engine.public_discard[c] - opp_cnt;
                rem = std::max(0, rem);
                for (int i = 0; i < rem; ++i) {
                    w.draw_pile.push_back(static_cast<uint8_t>(c));
                }
            }

            // Fisher-Yates shuffle draw pile with world k's seed
            std::mt19937_64 rng_k(seed + static_cast<uint64_t>(k) * 10007 + 1);
            std::shuffle(w.draw_pile.begin(), w.draw_pile.end(), rng_k);
        }

        // Step 2: Multi-threaded CRN setup & step root action
        {
            py::gil_scoped_release release;
            size_t n_workers = pool->size();
            std::vector<std::function<void()>> tasks;
            tasks.reserve(n_workers);

            for (size_t t = 0; t < n_workers; ++t) {
                int start = static_cast<int>(t * total_slots / n_workers);
                int end = static_cast<int>((t + 1) * total_slots / n_workers);

                tasks.emplace_back([this, start, end, &root_engine, &worlds, act_ptr, K, root_p, opp_p]() {
                    for (int slot = start; slot < end; ++slot) {
                        int a_idx = slot / K;
                        int k = slot % K;

                        auto& eng = engines[slot];
                        eng = root_engine;
                        eng.hands[opp_p] = worlds[k].opp_hand;
                        eng.deck = worlds[k].draw_pile;

                        acting_player_buf[slot] = root_p;
                        uint8_t a = static_cast<uint8_t>(act_ptr[a_idx]);
                        float r = eng.step(a);
                        rewards_buf[slot] = r;
                        dones_buf[slot] = eng.is_over ? 1 : 0;
                        _update_env_buffers(slot);
                    }
                });
            }
            pool->enqueue_batch(tasks);
            pool->wait_all();
            total_steps_.fetch_add(total_slots, std::memory_order_relaxed);
        }

        py::dict obs = get_obs_dict();
        py::array_t<float> py_rewards(num_envs, rewards_buf.data());
        py::array_t<bool> py_dones(num_envs, reinterpret_cast<const bool*>(dones_buf.data()));

        return py::make_tuple(obs, py_rewards, py_dones);
    }

    py::tuple step_indices(
        const py::array_t<int32_t, py::array::c_style | py::array::forcecast>& env_indices,
        const py::array_t<int32_t, py::array::c_style | py::array::forcecast>& actions
    ) {
        py::buffer_info idx_buf = env_indices.request();
        py::buffer_info act_buf = actions.request();
        int n_active = static_cast<int>(idx_buf.size);
        if (n_active != static_cast<int>(act_buf.size)) {
            throw std::runtime_error("env_indices size must match actions size");
        }
        const int32_t* idx_ptr = static_cast<const int32_t*>(idx_buf.ptr);
        const int32_t* act_ptr = static_cast<const int32_t*>(act_buf.ptr);

        {
            py::gil_scoped_release release;
            size_t n_workers = pool->size();
            std::vector<std::function<void()>> tasks;
            tasks.reserve(n_workers);

            for (size_t t = 0; t < n_workers; ++t) {
                int start = static_cast<int>(t * n_active / n_workers);
                int end = static_cast<int>((t + 1) * n_active / n_workers);

                tasks.emplace_back([this, start, end, idx_ptr, act_ptr]() {
                    for (int k = start; k < end; ++k) {
                        int i = idx_ptr[k];
                        auto& eng = engines[i];
                        acting_player_buf[i] = eng.current_player;
                        uint8_t a = static_cast<uint8_t>(act_ptr[k]);
                        float r = eng.step(a);
                        rewards_buf[i] = r;
                        dones_buf[i] = eng.is_over ? 1 : 0;
                        _update_env_buffers(i);
                    }
                });
            }
            pool->enqueue_batch(tasks);
            pool->wait_all();
            total_steps_.fetch_add(n_active, std::memory_order_relaxed);
        }

        py::dict obs = get_obs_dict();
        py::array_t<float> py_rewards(num_envs, rewards_buf.data());
        py::array_t<bool> py_dones(num_envs, reinterpret_cast<const bool*>(dones_buf.data()));

        return py::make_tuple(obs, py_rewards, py_dones);
    }

    uint64_t get_total_steps() const { return total_steps_.load(); }
    uint64_t get_total_episodes() const { return total_episodes_.load(); }
};

// ============================================================================
// Pybind11 Module Definition
// ============================================================================
PYBIND11_MODULE(_cpp, m) {
    m.doc() = "High-performance vectorized C++17 UNO simulator for uno-ataraxos";

    // Constants
    m.attr("NUM_ACTIONS") = NUM_ACTIONS;
    m.attr("NUM_CARD_TYPES") = NUM_CARD_TYPES;
    m.attr("TOTAL_DECK_CARDS") = TOTAL_DECK_CARDS;
    m.attr("OBS_DIM") = OBS_DIM;
    m.attr("MAX_TURNS") = MAX_TURNS;

    m.def("get_card_names", &get_card_names);
    m.def("get_action_names", &get_action_names);
    m.def("card_name_to_id", &card_name_to_id);
    m.def("action_name_to_id", &action_name_to_id);

    // UnoEngine
    py::class_<UnoEngine>(m, "UnoEngine")
        .def(py::init<uint64_t>(), py::arg("seed") = 42)
        .def("reset", &UnoEngine::reset, py::arg("seed") = 0)
        .def("clone", &UnoEngine::clone)
        .def("step", [](UnoEngine& self, const py::object& act) -> float {
            uint8_t a = 0;
            if (py::isinstance<py::str>(act)) {
                std::string s = act.cast<std::string>();
                int id = action_name_to_id(s);
                if (id < 0) throw std::runtime_error("Invalid action string: " + s);
                a = static_cast<uint8_t>(id);
            } else {
                a = static_cast<uint8_t>(act.cast<int>());
            }
            return self.step(a);
        }, py::arg("action"))
        .def("sync_from_sim", [](UnoEngine& self, const py::object& sim) {
            auto hands_obj = sim.attr("hands").cast<std::vector<std::vector<std::string>>>();
            auto h0 = hands_obj[0];
            auto h1 = hands_obj[1];
            auto d = sim.attr("deck").cast<std::vector<std::string>>();
            auto disc = sim.attr("discard_pile").cast<std::vector<std::string>>();
            auto tgt = sim.attr("target").cast<std::string>();
            int stk = sim.attr("stack").cast<int>();
            int curr = sim.attr("current_player").cast<int>();
            int turns = sim.attr("turn_count").cast<int>();
            int passes = sim.attr("consecutive_passes").cast<int>();

            int8_t act_col = 0;
            if (!tgt.empty()) {
                char c = tgt[0];
                if (c == 'r') act_col = 0;
                else if (c == 'g') act_col = 1;
                else if (c == 'b') act_col = 2;
                else if (c == 'y') act_col = 3;
            }
            self.load_state_from_strings(h0, h1, d, disc, tgt, act_col, stk, curr, turns, passes);
        }, py::arg("sim"))
        .def_property_readonly("winner", [](const UnoEngine& self) -> py::object {
            if (self.winner < 0) return py::none();
            return py::cast((int)self.winner);
        })
        .def("get_public_discard", &UnoEngine::get_public_discard)
        .def("get_last_reshuffle_pool", &UnoEngine::get_last_reshuffle_pool)
        .def("get_recycled_remaining_for_me", &UnoEngine::get_recycled_remaining_for_me, py::arg("player") = -1)
        .def("get_draws_since_reshuffle", &UnoEngine::get_draws_since_reshuffle, py::arg("player") = -1)
        .def("get_action_mask", [](const UnoEngine& self, int player) {
            py::array_t<bool> mask(NUM_ACTIONS);
            py::buffer_info buf = mask.request();
            self.get_action_mask(reinterpret_cast<uint8_t*>(buf.ptr), static_cast<int8_t>(player));
            return mask;
        }, py::arg("player") = -1)
        .def("get_legal_action_ids", &UnoEngine::get_legal_action_ids, py::arg("player") = -1)
        .def("get_legal_actions", &UnoEngine::get_legal_actions, py::arg("player") = -1)
        .def("get_payoffs", [](const UnoEngine& self) {
            auto p = self.get_payoffs();
            return py::make_tuple(p.first, p.second);
        })
        .def("get_available_draw_count", &UnoEngine::get_available_draw_count)
        .def("get_observation", [](const UnoEngine& self, int player) {
            py::array_t<float> vec(OBS_DIM);
            py::buffer_info buf = vec.request();
            self.encode_observation_vector(static_cast<float*>(buf.ptr), static_cast<int8_t>(player));
            return vec;
        }, py::arg("player") = -1)
        .def("get_card_features_actor", [](const UnoEngine& self, int player) {
            py::array_t<float> arr({NUM_CARD_TYPES, 7});
            py::buffer_info buf = arr.request();
            self.encode_card_features_actor(static_cast<float*>(buf.ptr), static_cast<int8_t>(player));
            return arr;
        }, py::arg("player") = -1)
        .def("get_global_features", [](const UnoEngine& self, int player) {
            py::array_t<float> arr(10);
            py::buffer_info buf = arr.request();
            self.encode_global_features(static_cast<float*>(buf.ptr), static_cast<int8_t>(player));
            return arr;
        }, py::arg("player") = -1)
        .def("get_history_tokens", [](const UnoEngine& self, int player) {
            py::array_t<int32_t> arr({48, 5});
            py::buffer_info buf = arr.request();
            self.encode_history_tokens(static_cast<int32_t*>(buf.ptr), static_cast<int8_t>(player));
            return arr;
        }, py::arg("player") = -1)
        .def("get_opp_hand_counts", [](const UnoEngine& self, int player) {
            py::array_t<float> arr(NUM_CARD_TYPES);
            py::buffer_info buf = arr.request();
            self.encode_opp_hand_counts(static_cast<float*>(buf.ptr), static_cast<int8_t>(player));
            return arr;
        }, py::arg("player") = -1)
        .def("load_state", &UnoEngine::load_state_from_strings,
             py::arg("hands_0"), py::arg("hands_1"), py::arg("deck"), py::arg("discard_pile"),
             py::arg("top_card"), py::arg("active_color"), py::arg("stack"),
             py::arg("current_player"), py::arg("turn_count"), py::arg("consecutive_passes"))
        .def("load_state_ids", [](UnoEngine& self,
                                  const std::vector<int>& h0,
                                  const std::vector<int>& h1,
                                  const std::vector<int>& deck,
                                  const std::vector<int>& discard,
                                  int top_card_act,
                                  int act_color,
                                  int stack,
                                  int current_player,
                                  int turn_count,
                                  int consecutive_passes) {
            std::vector<uint8_t> h0_u(h0.begin(), h0.end());
            std::vector<uint8_t> h1_u(h1.begin(), h1.end());
            std::vector<uint8_t> d_u(deck.begin(), deck.end());
            std::vector<uint8_t> disc_u(discard.begin(), discard.end());
            self.load_state(h0_u, h1_u, d_u, disc_u,
                            static_cast<uint8_t>(top_card_act),
                            static_cast<int8_t>(act_color),
                            stack,
                            static_cast<int8_t>(current_player),
                            turn_count,
                            consecutive_passes);
        }, py::arg("hands_0"), py::arg("hands_1"), py::arg("deck"), py::arg("discard_pile"),
           py::arg("top_card"), py::arg("active_color"), py::arg("stack"),
           py::arg("current_player"), py::arg("turn_count"), py::arg("consecutive_passes"))
        .def("set_world", [](UnoEngine& self, int opp_player, const std::vector<int>& opp_hand, const std::vector<int>& draw_pile) {
            std::vector<uint8_t> h(opp_hand.begin(), opp_hand.end());
            std::vector<uint8_t> d(draw_pile.begin(), draw_pile.end());
            self.set_world(static_cast<int8_t>(opp_player), h, d);
        }, py::arg("opp_player"), py::arg("opp_hand"), py::arg("draw_pile"))
        .def("get_hand_ids", &UnoEngine::get_hand_ids, py::arg("player"))
        .def("get_hands_str", &UnoEngine::get_hands_str, py::arg("player"))
        .def("get_deck_ids", &UnoEngine::get_deck_ids)
        .def("get_discard_pile_ids", &UnoEngine::get_discard_pile_ids)
        .def_readwrite("target", &UnoEngine::target)
        .def_readwrite("active_color", &UnoEngine::active_color)
        .def_readwrite("stack", &UnoEngine::stack)
        .def_readwrite("current_player", &UnoEngine::current_player)
        .def_readwrite("is_over", &UnoEngine::is_over)
        .def_readwrite("turn_count", &UnoEngine::turn_count)
        .def_readwrite("consecutive_passes", &UnoEngine::consecutive_passes)
        .def_property_readonly("target_str", &UnoEngine::get_target_str)
        .def_property_readonly("hands", [](const UnoEngine& self) {
            return py::make_tuple(self.get_hands_str(0), self.get_hands_str(1));
        })
        .def_property_readonly("deck", &UnoEngine::get_deck_str)
        .def_property_readonly("discard_pile", &UnoEngine::get_discard_pile_str)
        .def_property_readonly("played_cards", &UnoEngine::get_played_cards_str)
        .def("set_history_events", &UnoEngine::set_history_events, py::arg("events"))
        .def("set_reshuffle_provenance", &UnoEngine::set_reshuffle_provenance,
             py::arg("pool"), py::arg("d0"), py::arg("d1"), py::arg("k0"), py::arg("k1"));

    // VecUnoEnv
    py::class_<VecUnoEnv>(m, "VecUnoEnv")
        .def(py::init<int, uint64_t, int, bool>(),
             py::arg("num_envs"), py::arg("seed") = 42, py::arg("num_threads") = 0, py::arg("auto_reset") = true)
        .def("reset", &VecUnoEnv::reset)
        .def("step", &VecUnoEnv::step, py::arg("actions"))
        .def("step_no_autoreset", &VecUnoEnv::step_no_autoreset, py::arg("actions"))
        .def("step_random_legal", &VecUnoEnv::step_random_legal)
        .def("get_obs", &VecUnoEnv::get_obs_dict)
        .def("get_engine", [](VecUnoEnv& self, int i) -> UnoEngine& {
            if (i < 0 || i >= self.num_envs) throw std::out_of_range("env index out of range");
            return self.engines[i];
        }, py::return_value_policy::reference_internal, py::arg("index"))
        .def("load_worlds_and_step_root", &VecUnoEnv::load_worlds_and_step_root,
             py::arg("root_engine"), py::arg("sampled_opp_hands"), py::arg("legal_actions"), py::arg("seed") = 42)
        .def("step_indices", &VecUnoEnv::step_indices,
             py::arg("env_indices"), py::arg("actions"))
        .def("resize", &VecUnoEnv::resize, py::arg("new_num_envs"))
        .def_property_readonly("total_steps", &VecUnoEnv::get_total_steps)
        .def_property_readonly("total_episodes", &VecUnoEnv::get_total_episodes)
        .def_readonly("num_envs", &VecUnoEnv::num_envs);
}
