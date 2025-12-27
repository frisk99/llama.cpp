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
class mmtdClass{
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
    mmtdClass();
    virtual ~mmtdClass();
    bool load();
    void unload();
    std::string generate_tokens(std::string prompt , std::vector<std::string> image_path);
    void clear();
private:
    int generate_response(int n_predict);
    bool init_vision_context(common_params & params);
    bool check_antiprompt(const llama_tokens & generated_tokens);
    bool load_media(const std::string & fname);
    std::string chat_add_and_format(common_chat_msg & new_msg);
    int eval_message(common_chat_msg & msg);
};
