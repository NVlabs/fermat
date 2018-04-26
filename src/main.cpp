#include <buffers.h>
#include <ray.h>
#include <framebuffer.h>
#include <lights.h>
#include <renderer.h>

int main(int argc, char** argv)
{
	Renderer renderer;
	renderer.init(argc, argv);
	return 0;
}