nvcc collatz.cu -Xcudafe "--diag_suppress=integer_sign_change" -o collatz
collatz -s -M 5 -N 10000000000 -i 2147483648 -f col --showMem
pause