#include <unistd.h>
#include <iostream>
#include <cstdlib>
#include <signal.h>

namespace signal_handle
{
    void signal_callback_handler(int signum) {
        std::cout << "\033[34;1m" << "Caught signal "<<  signum << "\033[32;0m" <<std::endl;
        // Terminate program
        exit(signum);
    }
} // namespace signal_handle