#include <vector>

class BatchMemory{
    private:
        int              m_memory_size{0};
        int              m_batch_size{0};
        bool             m_full{false};
        std::vector<int> m_processed_batches;
        std::vector<int> m_memory_batch;

    public:

        BatchMemory() = default;
        
        BatchMemory(int memory_size, int batch_size)
            : m_memory_size(memory_size), m_batch_size(batch_size), m_full(false) {}


        // Getters
        int getMemorySize() const {return m_memory_size;}
        int getBatchSize() const {return m_batch_size;}
        bool getBufferStatus() const {return m_full;};
        
        const std::vector<int>& getProcessedBatches() const {return m_processed_batches;}
        const std::vector<int>& getMemoryBatch() const {return m_memory_batch;}

        // Setters
        void setMemorySize(int m_size) {m_memory_size = m_size;}
        void setBatchSize(int b_size) {m_batch_size = b_size;}
        void setBufferStatus(bool status) {m_full = status;}
        
        void resetBatch();
        void generateMiniBatches();

};