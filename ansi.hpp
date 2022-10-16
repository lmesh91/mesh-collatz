//This file defines some ANSI escape codes, for use in making fancy CLI output.

#include <string>
#include <windows.h>
#include <stdio.h>

struct ANSI {
    //For right now, I'm only defining the ones I might consider using.
    static constexpr const char* reset = "\x1B[0m";
    static constexpr const char* bold = "\x1B[1m";
    static constexpr const char* faint = "\x1B[2m";
    static constexpr const char* italic = "\x1B[3m";
    static constexpr const char* underline = "\x1B[4m";
    static constexpr const char* blink = "\x1B[5m";
    static constexpr const char* strikethrough = "\x1B[9m";
    static constexpr const char* doubleUnderline = "\x1B[21m";
    static constexpr const char* resetBold = "\x1B[22m";
    static constexpr const char* resetItalic = "\x1B[23m";
    static constexpr const char* resetUnderline = "\x1B[24m";
    static constexpr const char* resetBlink = "\x1B[25m";
    static constexpr const char* resetStrikethrough = "\x1B[29m";
    static constexpr const char* black = "\x1B[30m";
    static constexpr const char* red = "\x1B[31m";
    static constexpr const char* green = "\x1B[32m";
    static constexpr const char* yellow = "\x1B[33m";
    static constexpr const char* blue = "\x1B[34m";
    static constexpr const char* magenta = "\x1B[35m";
    static constexpr const char* cyan = "\x1B[36m";
    static constexpr const char* white = "\x1B[37m";
    static constexpr const char* brightBlack = "\x1B[90m";
    static constexpr const char* brightRed = "\x1B[91m";
    static constexpr const char* brightGreen = "\x1B[92m";
    static constexpr const char* brightYellow = "\x1B[93m";
    static constexpr const char* brightBlue = "\x1B[94m";
    static constexpr const char* brightMagenta = "\x1B[95m";
    static constexpr const char* brightCyan = "\x1B[96m";
    static constexpr const char* brightWhite = "\x1B[97m";
    static constexpr const char* resetColor = "\x1B[39m";
    static constexpr const char* blackBack = "\x1B[40m";
    static constexpr const char* redBack = "\x1B[41m";
    static constexpr const char* greenBack = "\x1B[42m";
    static constexpr const char* yellowBack = "\x1B[43m";
    static constexpr const char* blueBack = "\x1B[44m";
    static constexpr const char* magentaBack = "\x1B[45m";
    static constexpr const char* cyanBack = "\x1B[46m";
    static constexpr const char* whiteBack = "\x1B[47m";
    static constexpr const char* brightBlackBack = "\x1B[100m";
    static constexpr const char* brightRedBack = "\x1B[101m";
    static constexpr const char* brightGreenBack = "\x1B[102m";
    static constexpr const char* brightYellowBack = "\x1B[103m";
    static constexpr const char* brightBlueBack = "\x1B[104m";
    static constexpr const char* brightMagentaBack = "\x1B[105m";
    static constexpr const char* brightCyanBack = "\x1B[106m";
    static constexpr const char* brightWhiteBack = "\x1B[107m";
    static constexpr const char* resetColorBack = "\x1B[49m";
    static std::string customColor(int r, int g, int b) {
        return "\x1B[38;2;"+std::to_string(r)+";"+std::to_string(g)+";"+std::to_string(b);
    };
    static std::string customColorBack(int r, int g, int b) {
        return "\x1B[48;2;"+std::to_string(r)+";"+std::to_string(g)+";"+std::to_string(b);
    };

    static bool EnableVTMode() { //taken from Microsoft Docs
        // Set output mode to handle virtual terminal sequences
        HANDLE hOut = GetStdHandle(STD_OUTPUT_HANDLE);
        if (hOut == INVALID_HANDLE_VALUE)
        {
            return false;
        }

        DWORD dwMode = 0;
        if (!GetConsoleMode(hOut, &dwMode))
        {
            return false;
        }

        dwMode |= ENABLE_VIRTUAL_TERMINAL_PROCESSING;
        if (!SetConsoleMode(hOut, dwMode))
        {
            return false;
        }
        return true;
    };   
    private:
        ANSI() {}; //must be just static things
};