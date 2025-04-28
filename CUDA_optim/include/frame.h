#ifndef FRAME_H
#define FRAME_H

#include <cstdint>
#include <cstring>

enum FrameType { 
    I_frame, P_frame 
};

class Frame {
public:
    int width;
    int height;
    FrameType type;
    int channels;
    uint8_t *data;

    Frame(int w, int h, FrameType t, int ch = 1):width(w),height(h),type(t), channels(ch) {
        data = new uint8_t[w * h * ch];
        std::memset(data, 0, w * h * ch);
    }
    ~Frame(){ delete[] data;}

    Frame(Frame&& o) noexcept : width(o.width), height(o.height), type(o.type), channels(o.channels), data(o.data) { o.data=nullptr; }

    Frame& operator=(Frame&& o) noexcept {
        if(this!=&o){ delete[] data;
            width=o.width; height=o.height; type=o.type; channels = o.channels; data=o.data; o.data=nullptr; }
        return *this;
    }

    inline uint8_t  getPixel(int x,int y, int c = 0) const {
        return data[(y * width + x) * channels + c]; 
    }
    inline void setPixel(int x,int y,uint8_t v, int c = 0){ 
        data[(y * width + x) * channels + c] = v; 
    }
};

#endif  
