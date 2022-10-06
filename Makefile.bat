nvcc collatz.cu -Xcudafe "--diag_suppress=integer_sign_change" -o collatz
collatz -b
pause