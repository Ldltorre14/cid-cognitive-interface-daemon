#include <iostream>
#include <torch/torch.h> // Include LibTorch
#include "../include/memory/batch_memory.hpp"


int main() {
    std::cout << "This is a test" << std::endl;
    BatchMemory memory(10, 10);

    std::cout << memory.getMemorySize() << std::endl;
    // Quick test to ensure Torch is linked and working
    //torch::Tensor tensor = torch::rand({2, 3});
    //std::cout << tensor << std::endl;
    
    return 0;
}