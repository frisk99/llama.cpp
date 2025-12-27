#include "mtmd-class.h"
#include <iostream>
#include <vector>
#include <string>
#include <memory>

int main(int argc, char** argv) {

    //./test-mtmd-class  -p  "请帮我描述这张图" --image image_path
    std::cout << "Testing mtmd-class with smart pointer..." << std::endl;
    
    if (argc < 4) {
        std::cerr << "Usage: " << argv[0] << " <model_path> <mmproj_path> <prompt> [image_path1 image_path2 ...]" << std::endl;
        std::cerr << "Example: " << argv[0] << " models/llava-v1.5-7b.gguf models/llava-v1.5-7b-mmproj.gguf \"Describe this image\" image.jpg" << std::endl;
        return 1;
    }
    
    std::unique_ptr<mmtdClass> mtmd = std::make_unique<mmtdClass>();
    
    mtmd->argc = argc;
    mtmd->argv = argv;
    
    mtmd->params.model.path = "/home/stone/code/llama.cpp/qwen3-2b-f16.gguf";
    mtmd->params.mmproj.path = "/home/stone/code/llama.cpp/qwen3-2b-mmproj-f16.gguf";
    //mtmd->params.n_predict = 1024; // 生成最多50个token
    //mtmd->params.n_ctx = 4096;
    //mtmd->params.cpuparams.n_threads = 4;
    
    std::cout << "Loading model...111" << std::endl;
    if (!mtmd->load()) {
        std::cerr << "Failed to load model!" << std::endl;
        return 1;
    }
    
    std::cout << "Model loaded successfully!" << std::endl;
    
    std::string prompt = argv[2];
    std::vector<std::string> image_paths;
    std::string image = argv[4];
    image_paths.push_back(image);
    
    std::cout << "Generating response for prompt: \"" << prompt << "\"" << std::endl;
    if (!image_paths.empty()) {
        std::cout << "With " << image_paths.size() << " image(s): ";
        for (const auto& img : image_paths) {
            std::cout << img << " ";
        }
        std::cout << std::endl;
    }
    
    std::cout << "\n=== Generating tokens ===" << std::endl;
    auto start = std::chrono::steady_clock::now();
    std::string response = mtmd->generate_tokens(prompt, image_paths);
    auto end = std::chrono::steady_clock::now();
    auto diff = end - start;
    std::cout << "time cost: " 
              << std::chrono::duration_cast<std::chrono::milliseconds>(diff).count() << " ms ("
              << std::endl;
    if (response.empty()) {
        std::cerr << "Failed to generate response!" << std::endl;
        return 1;
    }
    


    std::cout << "\n=== Generated Response ===" << std::endl;
    std::cout << response << std::endl;
    std::cout << "==========================" << std::endl;
    

    mtmd->unload();
    
    std::cout << "\nTest completed successfully!" << std::endl;
    return 0;
}
