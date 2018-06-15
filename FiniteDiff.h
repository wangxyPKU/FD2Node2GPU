
extern "C" {
    void GetDeviceName();
    void GpuCalculate(float *fai, int M, int N, int my_rank, int comm_sz);
}
