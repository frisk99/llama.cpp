#include "arg.h"
#include "log.h"
#include "common.h"
#include "sampling.h"
#include "llama.h"
#include "ggml.h"
#include "console.h"
#include "chat.h"
#include "mtmd.h"
#include "mtmd-helper.h"

#include <vector>
#include <limits.h>
#include <cinttypes>

#if defined (__unix__) || (defined (__APPLE__) && defined (__MACH__))
#include <signal.h>
#include <unistd.h>
#elif defined (_WIN32)
#define WIN32_LEAN_AND_MEAN
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <signal.h>
#endif

// volatile, because of signal being an interrupt
static volatile bool g_is_generating = false;
static volatile bool g_is_interrupted = false;

/**
 * Please note that this is NOT a production-ready stuff.
 * It is a playground for trying multimodal support in llama.cpp.
 * For contributors: please keep this code simple and easy to understand.
 */

static void show_additional_info(int /*argc*/, char ** argv) {
    LOG(
        "Experimental CLI for multimodal\n\n"
        "Usage: %s [options] -m <model> --mmproj <mmproj> --image <image> --audio <audio> -p <prompt>\n\n"
        "  -m and --mmproj are required\n"
        "  -hf user/repo can replace both -m and --mmproj in most cases\n"
        "  --image, --audio and -p are optional, if NOT provided, the CLI will run in chat mode\n"
        "  to disable using GPU for mmproj model, add --no-mmproj-offload\n",
        argv[0]
    );
}

#if defined (__unix__) || (defined (__APPLE__) && defined (__MACH__)) || defined (_WIN32)
static void sigint_handler(int signo) {
    if (signo == SIGINT) {
        if (g_is_generating) {
            g_is_generating = false;
        } else {
            console::cleanup();
            if (g_is_interrupted) {
                _exit(1);
            }
            g_is_interrupted = true;
        }
    }
}
#endif

class MtmdCli {
public:
    int argc;
    char ** argv;

    common_params params;

    mtmd::context_ptr ctx_vision;
    common_init_result llama_init;

    llama_model       * model = nullptr;
    llama_context     * lctx = nullptr;
    const llama_vocab * vocab = nullptr;
    common_sampler    * smpl = nullptr;
    llama_batch         batch;
    int                 n_batch = 0;

    mtmd::bitmaps bitmaps;

    // chat template
    common_chat_templates_ptr tmpls;
    std::vector<common_chat_msg> chat_history;
    bool use_jinja = false;
    // TODO: support for --system-prompt with /clear command

    // support for legacy templates (models not having EOT token)
    llama_tokens antiprompt_tokens;

    int n_threads    = 1;
    llama_pos n_past = 0;

    bool loaded = false;

    MtmdCli(int argc, char ** argv) : argc(argc), argv(argv) {
        batch = {0};
    }

    ~MtmdCli() {
        unload();
    }

    bool load() {
        if (loaded) return true;

        params.sampling.temp = 0.2; // lower temp by default for better quality

        if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_MTMD, show_additional_info)) {
            return false;
        }

        common_init();
        mtmd_helper_log_set(common_log_default_callback, nullptr);

        if (params.mmproj.path.empty()) {
            show_additional_info(argc, argv);
            LOG_ERR("ERR: Missing --mmproj argument\n");
            return false;
        }

        llama_init = common_init_from_params(params);
        model = llama_init.model.get();
        lctx = llama_init.context.get();
        
        if (!model || !lctx) {
            return false;
        }

        vocab = llama_model_get_vocab(model);
        smpl = common_sampler_init(model, params.sampling);
        n_threads = params.cpuparams.n_threads;
        batch = llama_batch_init(1, 0, 1); // batch for next token generation
        n_batch = params.n_batch;

        if (!llama_model_chat_template(model, nullptr) && params.chat_template.empty()) {
            LOG_ERR("Model does not have chat template.\n");
            LOG_ERR("  For old llava models, you may need to use '--chat-template vicuna'\n");
            LOG_ERR("  For MobileVLM models, use '--chat-template deepseek'\n");
            LOG_ERR("  For Mistral Small 3.1, use '--chat-template mistral-v7'\n");
            return false;
        }

        tmpls = common_chat_templates_init(model, params.chat_template);
        use_jinja = params.use_jinja;
        chat_history.clear();
        LOG_INF("%s: chat template example:\n%s\n", __func__, common_chat_format_example(tmpls.get(), params.use_jinja, params.default_template_kwargs).c_str());

        if (!init_vision_context(params)) {
            return false;
        }

        // load antiprompt tokens for legacy templates
        if (params.chat_template == "vicuna") {
            antiprompt_tokens = common_tokenize(lctx, "ASSISTANT:", false, true);
        } else if (params.chat_template == "deepseek") {
            antiprompt_tokens = common_tokenize(lctx, "###", false, true);
        }

        loaded = true;
        return true;
    }

    void unload() {
        if (!loaded) return;

        if (smpl) {
            common_sampler_free(smpl);
            smpl = nullptr;
        }
        if (batch.token) {
            llama_batch_free(batch);
            batch = {0};
        }
        
        ctx_vision.reset();
        llama_init = common_init_result();
        model = nullptr;
        lctx = nullptr;
        vocab = nullptr;

        loaded = false;
    }

    bool inference(const std::string & prompt, const std::string & image_path) {
        if (!loaded) {
            LOG_ERR("Model not loaded\n");
            return false;
        }

        g_is_generating = true;
        
        std::string final_prompt = prompt;
        if (final_prompt.find(mtmd_default_marker()) == std::string::npos) {
            final_prompt += mtmd_default_marker();
        }

        if (!image_path.empty()) {
            if (!load_media(image_path)) {
                g_is_generating = false;
                return false;
            }
        }

        common_chat_msg msg;
        msg.role = "user";
        msg.content = final_prompt;

        if (eval_message(msg)) {
            g_is_generating = false;
            return false;
        }

        int n_predict = params.n_predict < 0 ? INT_MAX : params.n_predict;

        if (!g_is_interrupted && generate_response(n_predict)) {
            g_is_generating = false;
            return false;
        }

        g_is_generating = false;
        return true;
    }

    bool init_vision_context(common_params & params) {
        const char * clip_path = params.mmproj.path.c_str();
        mtmd_context_params mparams = mtmd_context_params_default();
        mparams.use_gpu          = params.mmproj_use_gpu;
        mparams.print_timings    = true;
        mparams.n_threads        = params.cpuparams.n_threads;
        mparams.flash_attn_type  = params.flash_attn_type;
        mparams.warmup           = params.warmup;
        mparams.image_min_tokens = params.image_min_tokens;
        mparams.image_max_tokens = params.image_max_tokens;
        ctx_vision.reset(mtmd_init_from_file(clip_path, model, mparams));
        if (!ctx_vision.get()) {
            LOG_ERR("Failed to load vision model from %s\n", clip_path);
            return false;
        }
        return true;
    }

    bool check_antiprompt(const llama_tokens & generated_tokens) {
        if (antiprompt_tokens.empty() || generated_tokens.size() < antiprompt_tokens.size()) {
            return false;
        }
        return std::equal(
            generated_tokens.end() - antiprompt_tokens.size(),
            generated_tokens.end(),
            antiprompt_tokens.begin()
        );
    }

    bool load_media(const std::string & fname) {
        mtmd::bitmap bmp(mtmd_helper_bitmap_init_from_file(ctx_vision.get(), fname.c_str()));
        if (!bmp.ptr) {
            return false;
        }
        bitmaps.entries.push_back(std::move(bmp));
        return true;
    }

    int generate_response(int n_predict) {
        llama_tokens generated_tokens;
        for (int i = 0; i < n_predict; i++) { // n_predict is max_tok
            if (i > n_predict || !g_is_generating || g_is_interrupted) {
                LOG("\n");
                break;
            }

            llama_token token_id = common_sampler_sample(smpl, lctx, -1);
            generated_tokens.push_back(token_id);
            common_sampler_accept(smpl, token_id, true);

            if (llama_vocab_is_eog(vocab, token_id) || check_antiprompt(generated_tokens)) {
                LOG("\n");
                break; // end of generation
            }

            LOG("%s", common_token_to_piece(lctx, token_id).c_str());
            fflush(stdout);

            if (g_is_interrupted) {
                LOG("\n");
                break;
            }

            // eval the token
            common_batch_clear(batch);
            common_batch_add(batch, token_id, n_past++, {0}, true);
            if (llama_decode(lctx, batch)) {
                LOG_ERR("failed to decode token\n");
                return 1;
            }
        }

        std::string generated_text = common_detokenize(lctx, generated_tokens);
        common_chat_msg msg;
        msg.role    = "assistant";
        msg.content = generated_text;
        chat_history.push_back(std::move(msg));

        return 0;
    }

    std::string chat_add_and_format(common_chat_msg & new_msg) {
        LOG_DBG("chat_add_and_format: new_msg.role='%s', new_msg.content='%s'\n",
            new_msg.role.c_str(), new_msg.content.c_str());
        auto formatted = common_chat_format_single(tmpls.get(), chat_history,
            new_msg, new_msg.role == "user",
            use_jinja);
        chat_history.push_back(new_msg);
        return formatted;
    }

    int eval_message(common_chat_msg & msg) {
        bool add_bos = chat_history.empty();
        auto formatted_chat = chat_add_and_format(msg);
        LOG_DBG("formatted_chat.prompt: %s\n", formatted_chat.c_str());

        mtmd_input_text text;
        text.text          = formatted_chat.c_str();
        text.add_special   = add_bos;
        text.parse_special = true;

        if (g_is_interrupted) return 0;

        mtmd::input_chunks chunks(mtmd_input_chunks_init());
        auto bitmaps_c_ptr = bitmaps.c_ptr();
        int32_t res = mtmd_tokenize(ctx_vision.get(),
                            chunks.ptr.get(), // output
                            &text, // text
                            bitmaps_c_ptr.data(),
                            bitmaps_c_ptr.size());
        if (res != 0) {
            LOG_ERR("Unable to tokenize prompt, res = %d\n", res);
            return 1;
        }

        bitmaps.entries.clear();

        llama_pos new_n_past;
        if (mtmd_helper_eval_chunks(ctx_vision.get(),
                    lctx, // lctx
                    chunks.ptr.get(), // chunks
                    n_past, // n_past
                    0, // seq_id
                    n_batch, // n_batch
                    true, // logits_last
                    &new_n_past)) {
            LOG_ERR("Unable to eval prompt\n");
            return 1;
        }

        n_past = new_n_past;

        LOG("\n");

        return 0;
    }
};

int main(int argc, char ** argv) {
    ggml_time_init();

    MtmdCli cli(argc, argv);
    if (!cli.load()) {
        return 1;
    }

    LOG_INF("%s: loading model: %s\n", __func__, cli.params.model.path.c_str());

    bool is_single_turn = !cli.params.prompt.empty() && !cli.params.image.empty();

    int n_predict = cli.params.n_predict < 0 ? INT_MAX : cli.params.n_predict;

    // Ctrl+C handling
    {
#if defined (__unix__) || (defined (__APPLE__) && defined (__MACH__))
        struct sigaction sigint_action;
        sigint_action.sa_handler = sigint_handler;
        sigemptyset (&sigint_action.sa_mask);
        sigint_action.sa_flags = 0;
        sigaction(SIGINT, &sigint_action, NULL);
#elif defined (_WIN32)
        auto console_ctrl_handler = +[](DWORD ctrl_type) -> BOOL {
            return (ctrl_type == CTRL_C_EVENT) ? (sigint_handler(SIGINT), true) : false;
        };
        SetConsoleCtrlHandler(reinterpret_cast<PHANDLER_ROUTINE>(console_ctrl_handler), true);
#endif
    }

    if (g_is_interrupted) return 130;

    if (is_single_turn) {
        g_is_generating = true;
        if (cli.params.prompt.find(mtmd_default_marker()) == std::string::npos) {
            for (size_t i = 0; i < cli.params.image.size(); i++) {
                cli.params.prompt += mtmd_default_marker();
            }
        }
        common_chat_msg msg;
        msg.role = "user";
        msg.content = cli.params.prompt;
        for (const auto & image : cli.params.image) {
            if (!cli.load_media(image)) {
                return 1; // error is already printed by libmtmd
            }
        }
        if (cli.eval_message(msg)) {
            return 1;
        }
        if (!g_is_interrupted && cli.generate_response(n_predict)) {
            return 1;
        }

    } else {
        LOG("\n Running in chat mode, available commands:");
        if (mtmd_support_vision(cli.ctx_vision.get())) {
            LOG("\n   /image <path>    load an image");
        }
        if (mtmd_support_audio(cli.ctx_vision.get())) {
            LOG("\n   /audio <path>    load an audio");
        }
        LOG("\n   /clear           clear the chat history");
        LOG("\n   /quit or /exit   exit the program");
        LOG("\n");

        std::string content;
        //this is generate 
        while (!g_is_interrupted) {
            g_is_generating = false;
            LOG("\n> ");
            console::set_display(console::user_input);
            std::string line;
            console::readline(line, false);
            if (g_is_interrupted) break;
            console::set_display(console::reset);
            line = string_strip(line);
            if (line.empty()) {
                continue;
            }
            if (line == "/quit" || line == "/exit") {
                break;
            }
            if (line == "/clear") {
                cli.n_past = 0;
                cli.chat_history.clear();
                llama_memory_clear(llama_get_memory(cli.lctx), true);
                LOG("Chat history cleared\n\n");
                continue;
            }
            g_is_generating = true;
            bool is_image = line == "/image" || line.find("/image ") == 0;
            bool is_audio = line == "/audio" || line.find("/audio ") == 0;
            if (is_image || is_audio) {
                if (line.size() < 8) {
                    LOG_ERR("ERR: Missing media filename\n");
                    continue;
                }
                std::string media_path = line.substr(7);
                if (cli.load_media(media_path)) {
                    LOG("%s %s loaded\n", media_path.c_str(), is_image ? "image" : "audio");
                    content += mtmd_default_marker();
                }
                // else, error is already printed by libmtmd
                continue;
            } else {
                content += line;
            }
            common_chat_msg msg;
            msg.role = "user";
            msg.content = content;
            int ret = cli.eval_message(msg);
            if (ret) {
                return 1;
            }
            if (g_is_interrupted) break;
            if (cli.generate_response(n_predict)) {
                return 1;
            }
            content.clear();
        }
    }
    if (g_is_interrupted) LOG("\nInterrupted by user\n");
    LOG("\n\n");
    llama_perf_context_print(cli.lctx);
    return g_is_interrupted ? 130 : 0;
}
