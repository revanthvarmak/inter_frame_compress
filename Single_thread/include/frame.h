#ifndef FRAME_H
#define FRAME_H

#include <vector>
#include <cstdint>
#include <cstddef>
#include <cstring>

enum FrameType{
    I_frame,P_frame
};

class Frame{
    public:
        int width;
        int height;
        uint8_t *data;  
        FrameType type;
        int channels;
        Frame(int w, int h, FrameType type, int ch = 1):width(w), height(h), type(type), channels(ch){
            data = new uint8_t[size_t(w)*h*channels];
            std::memset(data, 0, size_t(w)*h*channels);
        }
        ~Frame(){ 
            delete[] data;
        }
        
        Frame(const Frame&) = delete;
        Frame& operator=(const Frame&) = delete;

        Frame(Frame&& o) noexcept
          : width  (o.width),
            height (o.height),
            type   (o.type),
            channels(o.channels),
            data   (o.data)
        {
            o.data = nullptr;
        }
        Frame& operator=(Frame&& o) noexcept {
            if (this != &o) {
                delete[] data;
                width    = o.width;
                height   = o.height;
                type     = o.type;
                channels = o.channels;
                data     = o.data;
                o.data   = nullptr;
            }
            return *this;
   }


        uint8_t getPixel(int x, int y, int c) const{
            if(x < 0 || x >= width || y < 0 || y >= height || c < 0 || c >= channels) return 0;
            return data[((y * width + x) * channels) + c];
        }

        void setPixel(int x, int y, int c, uint8_t value){
            if(x < 0 || x >= width || y < 0 || y >= height || c < 0 || c >= channels) return;
            data[((y * width + x) * channels) + c] = value;
        }

};

#endif