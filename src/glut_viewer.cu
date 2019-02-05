/*
 * Fermat
 *
 * Copyright (c) 2016-2019, NVIDIA CORPORATION. All rights reserved.
 * 
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *    * Redistributions of source code must retain the above copyright
 *      notice, this list of conditions and the following disclaimer.
 *    * Redistributions in binary form must reproduce the above copyright
 *      notice, this list of conditions and the following disclaimer in the
 *      documentation and/or other materials provided with the distribution.
 *    * Neither the name of the NVIDIA CORPORATION nor the
 *      names of its contributors may be used to endorse or promote products
 *      derived from this software without specific prior written permission.
 * 
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE FOR ANY
 * DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#include <glut_viewer.h>
#include <optixu/optixu_matrix.h>
#include <cugar/linalg/vector.h>
#include <cugar/linalg/matrix.h>
#include <cugar/image/tga.h>
#include <cugar/sampling/random.h>
#include <cugar/sampling/distributions.h>
#include <cugar/sampling/variance.h>

#define USE_FFMPEG_HAXXX

GlutViewer* s_renderer;

uint8* copy_rgba_target(DomainBuffer<HOST_BUFFER,uint8>& h_rgba)
{
	const uint32 rgba_bytes = s_renderer->res().x * s_renderer->res().y * 4u;
	h_rgba.alloc( rgba_bytes );
	h_rgba.copy_from( rgba_bytes, CUDA_BUFFER, s_renderer->get_device_rgba_buffer() );
	return h_rgba.ptr();
}

// Renderer initialization
//
void GlutViewer::init(int argc, char** argv)
{
	strcpy(m_output_name, "output");
	m_orientation = Y_UP;

	m_record   = false;
	m_playback = false;
	m_accumulation = true;

	int encode_video = 0;

	for (int i = 0; i < argc; ++i)
	{
		if (strcmp(argv[i], "-X") == 0)
			m_orientation = X_UP;
		else if (strcmp(argv[i], "-Y") == 0)
			m_orientation = Y_UP;
		else if (strcmp(argv[i], "-Z") == 0)
			m_orientation = Z_UP;
		else if (strcmp(argv[i], "-o") == 0)
			strcpy(m_output_name, argv[++i]);
		else if (strcmp(argv[i], "-video") == 0)
			encode_video = atoi(argv[++i]);
		else if (strcmp(argv[i], "-record") == 0)
			m_record = true;
		else if (strcmp(argv[i], "-playback") == 0)
			m_playback = true;
		else if (strcmp(argv[i], "-accumulation") == 0)
			m_accumulation = atoi(argv[++i]);
	}

	if (m_orientation == Z_UP)
	{
		get_camera().eye = make_float3(0.0f,-1.0f, 0.0f);
		get_camera().aim = make_float3(0.0f, 0.0f, 0.0f);
		get_camera().up  = make_float3(0.0f, 0.0f, 1.0f);
		get_camera().dx  = normalize(cross(get_camera().aim - get_camera().eye, get_camera().up));
	}
	else if (m_orientation == X_UP)
	{
		get_camera().eye = make_float3(0.0f, 0.0f,-1.0f);
		get_camera().aim = make_float3(0.0f, 0.0f, 0.0f);
		get_camera().up  = make_float3(1.0f, 0.0f, 0.0f);
		get_camera().dx  = normalize(cross(get_camera().aim - get_camera().eye, get_camera().up));
	}
	else
	{
		get_camera().eye = make_float3(0.0f, 0.0f,-1.0f);
		get_camera().aim = make_float3(0.0f, 0.0f, 0.0f);
		get_camera().up  = make_float3(0.0f, 1.0f, 0.0f);
		get_camera().dx  = normalize(cross(get_camera().aim - get_camera().eye, get_camera().up));
	}

	s_renderer->RenderingContext::init(argc, argv);

	m_dollying = m_walking = m_panning = m_zooming = m_light_rotation = m_moving_selection = false;

	m_selected_group = uint32(-1);

	m_dirty_geo = false;
	m_dirty		= true;

	m_instance = 0;

  #ifdef USE_FFMPEG_HAXXX
	if (encode_video)
	{
		// start ffmpeg telling it to expect raw rgba 512x512p-30hz frames
		// -i - tells it to read frames from stdin
		char cmd[2048];
		sprintf(cmd,
			"ffmpeg -r %u -f rawvideo -pix_fmt rgba -s %ux%u -i - "
			"-threads 0 -preset fast -y -pix_fmt yuv420p -crf 21 -vf vflip output.mp4",
			encode_video,
			s_renderer->get_res().x, s_renderer->get_res().y);
		fprintf(stderr, "encode: %s\n", cmd);
		// open pipe to ffmpeg's stdin in binary write mode
		m_ffmpeg = _popen(cmd, "wb");
	}
	else
		m_ffmpeg = NULL;
  #endif

	if (m_record)
		m_camera_path = fopen("camera-path.txt", "w");
	else if (m_playback)
		m_camera_path = fopen("camera-path.txt", "r");
	else
		m_camera_path = NULL;

	s_renderer->clear();
	//s_renderer->render(0);
	
	glutInit(&argc, argv);//Initialize GLUT
	glutInitWindowSize(res().x, res().y);//define the window size
	glutInitWindowPosition(10, 50);//Position the window
	glutInitDisplayMode(GLUT_SINGLE | GLUT_RGBA | GLUT_DEPTH);//Define the drawing mode
	glutCreateWindow("Fermat");//Create our window

	glDepthMask(GL_TRUE);
	glDepthFunc(GL_LESS);
	glEnable(GL_DEPTH_TEST);

	gluOrtho2D(-1.0f, 1.0f, -1.0f, 1.0f);

	glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
	glPixelStorei(GL_PACK_ALIGNMENT, 1);
	glGenTextures(1, &m_texture);   // generate a texture handler really recommended (mandatory in openGL 3.0)

	glutDisplayFunc(GlutViewer::display);
	glutKeyboardFunc(GlutViewer::keyboard);
	glutMouseFunc(GlutViewer::mouse);
	glutMotionFunc(GlutViewer::motion);
	glutIdleFunc(GlutViewer::idle);
	glutMainLoop();//Keep the program running
}

// Renderer display function
//
void GlutViewer::display()
{
	if (s_renderer->m_dirty_geo)
	{
		//fprintf(stderr, "geometry update: started\n");
		s_renderer->update_model();
		s_renderer->m_dirty_geo = false;
		//fprintf(stderr, "geometry update: done\n");
	}

	if (s_renderer->m_dirty)
	{
		s_renderer->m_dirty = false;
		s_renderer->m_instance = 0;
		s_renderer->clear();
		//s_renderer->render(s_renderer->m_instance++);
	}

	if (s_renderer->m_record && s_renderer->m_camera_path)
	{
		fprintf(s_renderer->m_camera_path, "  %f %f %f\n", s_renderer->get_camera().eye.x, s_renderer->get_camera().eye.y, s_renderer->get_camera().eye.z);
		fprintf(s_renderer->m_camera_path, "  %f %f %f\n", s_renderer->get_camera().aim.x, s_renderer->get_camera().aim.y, s_renderer->get_camera().aim.z);
		fprintf(s_renderer->m_camera_path, "  %f %f %f\n", s_renderer->get_camera().up.x, s_renderer->get_camera().up.y, s_renderer->get_camera().up.z);
		fprintf(s_renderer->m_camera_path, "  %f %f %f\n", s_renderer->get_camera().dx.x, s_renderer->get_camera().dx.y, s_renderer->get_camera().dx.z);
		fprintf(s_renderer->m_camera_path, "  %f\n", s_renderer->get_camera().fov);
		fprintf(s_renderer->m_camera_path, "  %u\n\n", s_renderer->m_instance);
	}
	else if (s_renderer->m_playback && s_renderer->m_camera_path)
	{
		if (fscanf(s_renderer->m_camera_path, "  %f %f %f\n", &s_renderer->get_camera().eye.x, &s_renderer->get_camera().eye.y, &s_renderer->get_camera().eye.z) < 3) exit(0);
		if (fscanf(s_renderer->m_camera_path, "  %f %f %f\n", &s_renderer->get_camera().aim.x, &s_renderer->get_camera().aim.y, &s_renderer->get_camera().aim.z) < 3) exit(0);
		if (fscanf(s_renderer->m_camera_path, "  %f %f %f\n", &s_renderer->get_camera().up.x, &s_renderer->get_camera().up.y, &s_renderer->get_camera().up.z) < 3) exit(0);
		if (fscanf(s_renderer->m_camera_path, "  %f %f %f\n", &s_renderer->get_camera().dx.x, &s_renderer->get_camera().dx.y, &s_renderer->get_camera().dx.z) < 3) exit(0);
		if (fscanf(s_renderer->m_camera_path, "  %f\n", &s_renderer->get_camera().fov) < 1) exit(0);
		if (fscanf(s_renderer->m_camera_path, "  %u\n\n", &s_renderer->m_instance) < 1) exit(0);
		if (s_renderer->m_instance == 0)
			s_renderer->clear();
	}

	s_renderer->render(s_renderer->m_instance);

	if (s_renderer->m_accumulation)
		s_renderer->m_instance++;

	DomainBuffer<HOST_BUFFER,uint8> h_rgba;
	const uint8* rgba = copy_rgba_target( h_rgba );

	if (s_renderer->m_instance >= 32 && cugar::is_pow2(s_renderer->m_instance))
	{
		// dump the image to a tga
		char filename[1024];
		sprintf(filename, "%s-%u.tga", s_renderer->m_output_name, s_renderer->m_instance);
		fprintf(stderr, "saving %s\n", filename);

		cugar::write_tga(filename, s_renderer->get_res().x, s_renderer->get_res().y, rgba, cugar::TGAPixels::RGBA);
	}

	// reset the projection matrix
	glLoadIdentity();
	gluOrtho2D(-1.0f, 1.0f, -1.0f, 1.0f);

	glBindTexture(GL_TEXTURE_2D, s_renderer->m_texture); // tell openGL that we are using the texture
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);     //set our filter
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);     //set our filter
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB8, s_renderer->get_res().x, s_renderer->get_res().y, 0, GL_RGBA, GL_UNSIGNED_BYTE, (GLvoid*)&rgba[0]); // send the texture data
	
	glClearColor(0, 0, 0, 0);
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);  //clears the colour and depth buffers

	glEnable(GL_TEXTURE_2D); // you should use shader, but for an example fixed pipeline is ok ;)
	glBindTexture(GL_TEXTURE_2D, s_renderer->m_texture);

	// reset the base color to white
	glColor3f(1.0f, 1.0f, 1.0f);

	glBegin(GL_TRIANGLE_STRIP);  // draw something with the texture on
	glTexCoord2f(0.0, 0.0);
	glVertex2f(-1.0, -1.0);

	glTexCoord2f(1.0, 0.0);
	glVertex2f(1.0, -1.0);

	glTexCoord2f(0.0, 1.0);
	glVertex2f(-1.0, 1.0);

	glTexCoord2f(1.0, 1.0);
	glVertex2f(1.0, 1.0);
	glEnd();

	glClear(GL_DEPTH_BUFFER_BIT);  //clears the colour and depth buffers

	s_renderer->get_renderer()->draw( *s_renderer );

	glFlush(); //Draw everything to the screen

  #ifdef USE_FFMPEG_HAXXX
	if (s_renderer->m_ffmpeg)
		fwrite(&rgba[0], sizeof(int)*s_renderer->get_res().x*s_renderer->get_res().y, 1, s_renderer->m_ffmpeg);
  #endif
}

void GlutViewer::keyboard(unsigned char character, int x, int y)
{
	switch (character)
	{
	case '#':
		s_renderer->m_accumulation = !s_renderer->m_accumulation;
		s_renderer->m_dirty = true;
		glutPostRedisplay();
		break;
	case 'c':
		// dump the camera out
		fprintf(stderr, "\n\ncamera:\n");
		fprintf(stderr, "  %f %f %f\n", s_renderer->get_camera().eye.x, s_renderer->get_camera().eye.y, s_renderer->get_camera().eye.z);
		fprintf(stderr, "  %f %f %f\n", s_renderer->get_camera().aim.x, s_renderer->get_camera().aim.y, s_renderer->get_camera().aim.z);
		fprintf(stderr, "  %f %f %f\n", s_renderer->get_camera().up.x, s_renderer->get_camera().up.y, s_renderer->get_camera().up.z);
		fprintf(stderr, "  %f\n\n", s_renderer->get_camera().fov);
        break;
	//case 'g':
	//	fprintf(stderr, "select group: ");
	//	fscanf(stdin, "%u", &s_renderer->m_selected_group);
	//	break;
	case 'r':
        s_renderer->get_shading_mode() = kShaded;
		glutPostRedisplay();
        break;
	case 'f':
		s_renderer->get_shading_mode() = kFiltered;
		glutPostRedisplay();
		break;
	case 'v':
		s_renderer->get_shading_mode() = kVariance;
		glutPostRedisplay();
		break;
	case 'a':
		s_renderer->get_shading_mode() = kAlbedo;
		glutPostRedisplay();
		break;
	case 'l':
		s_renderer->get_shading_mode() = kDirectLighting;
		glutPostRedisplay();
		break;
	case 'd':
		s_renderer->get_shading_mode() = kDiffuseColor;
		glutPostRedisplay();
		break;
	case 'D':
		s_renderer->get_shading_mode() = kDiffuseAlbedo;
		glutPostRedisplay();
		break;
	case 'n':
        s_renderer->get_shading_mode() = kNormal;
		glutPostRedisplay();
        break;
	case 's':
		s_renderer->get_shading_mode() = kSpecularColor;
		glutPostRedisplay();
		break;
	case 'S':
		s_renderer->get_shading_mode() = kSpecularAlbedo;
		glutPostRedisplay();
		break;
	case 'u':
        s_renderer->get_shading_mode() = kUV;
		glutPostRedisplay();
        break;
    case 'U':
        s_renderer->get_shading_mode() = kUVStretch;
		glutPostRedisplay();
        break;
    case 'C':
        s_renderer->get_shading_mode() = kCharts;
		glutPostRedisplay();
        break;
	case 'R':
	{
		s_renderer->clear();
		for (uint32 i = 0; i < 64; ++i)
		{
			s_renderer->render(i);
			if (cugar::is_pow2(i+1))
			{
				// dump the image to a tga
				DomainBuffer<HOST_BUFFER,uint8> h_rgba;
				const uint8* rgba = copy_rgba_target( h_rgba );

				char filename[1024];
				sprintf(filename, "%s-%u.tga", s_renderer->m_output_name, i);

				cugar::write_tga(filename, s_renderer->get_res().x, s_renderer->get_res().y, rgba, cugar::TGAPixels::RGBA);
			}
		}
		break;
	}
	case '+':
		s_renderer->set_exposure( 2.0f * s_renderer->get_exposure() );
		glutPostRedisplay();
		break;
	case '-':
		s_renderer->set_exposure( 0.5f * s_renderer->get_exposure() );
		glutPostRedisplay();
		break;
	case 'G':
		s_renderer->set_gamma( 1.1f * s_renderer->get_gamma() );
		glutPostRedisplay();
		break;
	case 'g':
		s_renderer->set_gamma( (1.0f / 1.1f) * s_renderer->get_gamma() );
		glutPostRedisplay();
		break;
	case 'q':
	  #ifdef USE_FFMPEG_HAXXX
		if (s_renderer->m_ffmpeg)
			_pclose(s_renderer->m_ffmpeg);
	  #endif
		exit(0);
		break;
	case '1':
        s_renderer->get_shading_mode() = kAux0;
		glutPostRedisplay();
		break;
	case '2':
        s_renderer->get_shading_mode() = ShadingMode( kAux0 + 1 );
		glutPostRedisplay();
		break;
	case '3':
        s_renderer->get_shading_mode() = ShadingMode( kAux0 + 2 );
		glutPostRedisplay();
		break;
	case '!':
	{
		DomainBuffer<HOST_BUFFER,uint8> h_rgba;
		const uint8* rgba = copy_rgba_target( h_rgba );

		// dump the image to a tga
		char filename[1024];
		sprintf(filename, "%s-%u.tga", s_renderer->m_output_name, s_renderer->m_instance);
		fprintf(stderr, "saving %s\n", filename);

		cugar::write_tga(filename, s_renderer->get_res().x, s_renderer->get_res().y, rgba, cugar::TGAPixels::RGBA);
	}
	default:
		s_renderer->get_renderer()->keyboard(character, x, y, s_renderer->m_dirty);
		if (s_renderer->m_dirty)
			glutPostRedisplay();
	}
}

// Renderer idle function
//
void GlutViewer::idle()
{
	glutPostRedisplay();
}

void GlutViewer::mouse(int button, int state, int x, int y)
{
	// correct the pixel coordinates relative to the actual render target
	const uint2 res = make_uint2(s_renderer->get_res().x, s_renderer->get_res().y);

	const float win_w = glutGet(GLUT_WINDOW_WIDTH);
	const float win_h = glutGet(GLUT_WINDOW_HEIGHT) - 10; // seems the window is really 10 pixels less than what gets reported

	const uint32 px = uint32(res.x * (float(x)/win_w) );
	const uint32 py = uint32(cugar::max(cugar::min((res.y-9) * (float(win_h - y - 1)/win_h), float(res.y)-1),0.0f)); // looks like the window title bar covers about 9 pixels

	s_renderer->get_renderer()->mouse( *s_renderer, button, state, px, py );

	const bool ctrl  = (glutGetModifiers() & GLUT_ACTIVE_CTRL)  ? true : false;
	const bool shift = (glutGetModifiers() & GLUT_ACTIVE_SHIFT) ? true : false;

	if (button == GLUT_LEFT_BUTTON)
	{
		if (shift)
		{
			if (state == GLUT_DOWN)
			{
				if (x >= res.x || y >= res.y)
					return;

				const uint32 pixel = x + (res.y - y - 1)*res.x;

				// fetch the triangle id from the gbuffer
				const uint32 tri_id = s_renderer->get_frame_buffer().gbuffer.tri[pixel];

				if (tri_id != 0xFFFFFFFF)
				{
					s_renderer->m_moving_selection = true;
					s_renderer->m_mouse.x = x;
					s_renderer->m_mouse.y = y;

					// find the group containing this triangle id
					const uint32 group_id = cugar::upper_bound_index(tri_id, s_renderer->get_host_mesh().getGroupOffsets() + 1, s_renderer->get_host_mesh().getNumGroups());

					fprintf(stderr, "selected tri[%u] -> group[%u:%s]\n", tri_id, group_id, s_renderer->get_host_mesh().getGroupName(group_id).c_str());
					s_renderer->m_selected_group = group_id;

					// readback the gbuffer depth
					//const float depth = s_renderer->get_frame_buffer().gbuffer.depth[pixel];
					const float4 packed_geo = s_renderer->get_frame_buffer().gbuffer.geo[pixel];
					const cugar::Vector3f dir = GBufferView::unpack_pos(packed_geo) - s_renderer->get_camera().eye;

					cugar::Vector3f U, V, W;
					camera_frame( s_renderer->get_camera(), s_renderer->get_aspect_ratio(), U, V, W );

					//s_renderer->m_mouse_depth = depth / cugar::length(W);
					s_renderer->m_mouse_depth = dot(dir,W) / cugar::square_length(W);

					fprintf(stderr, "mouse depth: %f (%f)\n", s_renderer->m_mouse_depth, length(W));
				}
			}
			else
				s_renderer->m_moving_selection = true;
		}
		else
		{
			if (state == GLUT_DOWN)
			{
				s_renderer->m_dollying = true;
				s_renderer->m_mouse.x = x;
				s_renderer->m_mouse.y = y;
				s_renderer->m_camera_o = s_renderer->get_camera();

				fprintf(stderr, "mouse at %u, %u:\n", px, py);

				const uint32 pixel_idx = px + py*res.x;
				cugar::Vector4f diffuse_albedo  = s_renderer->get_frame_buffer().channels[FBufferDesc::DIFFUSE_A](pixel_idx);
				cugar::Vector4f specular_albedo = s_renderer->get_frame_buffer().channels[FBufferDesc::SPECULAR_A](pixel_idx);
				cugar::Vector4f diffuse_color   = s_renderer->get_frame_buffer().channels[FBufferDesc::DIFFUSE_C](pixel_idx);
				cugar::Vector4f specular_color  = s_renderer->get_frame_buffer().channels[FBufferDesc::SPECULAR_C](pixel_idx);
				fprintf(stderr, "  diffuse  : (%f, %f, %f) (var: %f) =\n             (%f, %f, %f) *\n             (%f, %f, %f)\n",
					diffuse_albedo.x * diffuse_color.x, diffuse_albedo.y * diffuse_color.y, diffuse_albedo.z * diffuse_color.z,
					diffuse_color.w,
					diffuse_albedo.x, diffuse_albedo.y, diffuse_albedo.z,
					diffuse_color.x, diffuse_color.y, diffuse_color.z);
				fprintf(stderr, "  specular : (%f, %f, %f) (var: %f) =\n             (%f, %f, %f) *\n             (%f, %f, %f)\n",
					specular_albedo.x * specular_color.x, specular_albedo.y * specular_color.y, specular_color.z * specular_color.z,
					specular_color.w,
					specular_albedo.x, specular_albedo.y, specular_albedo.z,
					specular_color.x, specular_color.y, specular_color.z);
			}
			else
				s_renderer->m_dollying = false;
		}
	}
	else if (button == GLUT_RIGHT_BUTTON)
	{
		if (ctrl)
		{
			if (state == GLUT_DOWN)
			{
				s_renderer->m_walking = true;
				s_renderer->m_mouse.x = x;
				s_renderer->m_mouse.y = y;
				s_renderer->m_camera_o = s_renderer->get_camera();
			}
			else
				s_renderer->m_walking = false;
		}
		else
		{
			if (state == GLUT_DOWN)
			{
				s_renderer->m_panning = true;
				s_renderer->m_mouse.x = x;
				s_renderer->m_mouse.y = y;
				s_renderer->m_camera_o = s_renderer->get_camera();
			}
			else
				s_renderer->m_panning = false;
		}
	}
	else if (button == GLUT_MIDDLE_BUTTON)
	{
		if (ctrl)
		{
			if (state == GLUT_DOWN)
			{
				s_renderer->m_light_rotation = true;
				s_renderer->m_mouse.x = x;
				s_renderer->m_mouse.y = y;
				if (s_renderer->get_directional_light_count()) // NOTE: use the FIRST directional light
					s_renderer->m_light_o = s_renderer->get_host_directional_lights()[0];
			}
			else
				s_renderer->m_light_rotation = false;
		}
		else
		{
			if (state == GLUT_DOWN)
			{
				s_renderer->m_zooming = true;
				s_renderer->m_mouse.x = x;
				s_renderer->m_mouse.y = y;
				s_renderer->m_camera_o = s_renderer->get_camera();
			}
			else
				s_renderer->m_zooming = false;
		}
	}
}
void GlutViewer::motion(int x, int y)
{
	if (s_renderer->m_dollying)
	{
		const float fdx = (x - s_renderer->m_mouse.x) / 160.0f;
		const float fdy = (y - s_renderer->m_mouse.y) / 160.0f;

		s_renderer->get_camera() = s_renderer->m_camera_o.rotate(make_float2(fdy, fdx));

		s_renderer->m_dirty = true;
	}
	else if (s_renderer->m_panning)
	{
		s_renderer->get_camera() = s_renderer->m_camera_o.pan(
			make_float2(float(x - s_renderer->m_mouse.x) / 800.0f,
						float(y - s_renderer->m_mouse.y) / 800.0f));

		s_renderer->m_dirty = true;
	}
	else if (s_renderer->m_walking)
	{
		s_renderer->get_camera() = s_renderer->m_camera_o.walk(
			float(y - s_renderer->m_mouse.y) / 800.0f);

		s_renderer->m_dirty = true;
	}
	else if (s_renderer->m_zooming)
	{
		s_renderer->get_camera() = s_renderer->m_camera_o.zoom(
			-float(y - s_renderer->m_mouse.y) / 100.0f);

		s_renderer->m_dirty = true;
	}
	else if (s_renderer->m_light_rotation)
	{
		DirectionalLight light = s_renderer->m_light_o;

		cugar::Vector2f rot(
			float(x - s_renderer->m_mouse.x) / 160.0f,
			float(y - s_renderer->m_mouse.y) / 160.0f);

		// rotate around the camera's axes
		const cugar::Vector3f dz = s_renderer->get_camera().up;

		const cugar::Matrix4x4f rot_X = cugar::rotation_around_axis(rot.y, dz);
		const cugar::Matrix4x4f rot_Y = cugar::rotation_around_X(rot.x);
 
		light.dir = cugar::vtrans( rot_Y, cugar::vtrans( rot_X, light.dir ) );

		if (s_renderer->get_directional_light_count()) // NOTE: use the FIRST directional light
		{
			s_renderer->set_directional_light(0, light);
			s_renderer->m_light_o = light;
		}

		s_renderer->m_dirty_geo = true;
		s_renderer->m_dirty		= true;
	}
	else if (s_renderer->m_moving_selection && s_renderer->m_selected_group != uint32(-1))
	{
		const uint2 res = make_uint2(s_renderer->get_res().x, s_renderer->get_res().y);

		const int32 mx = s_renderer->m_mouse.x;
		const int32 my = s_renderer->m_mouse.y;

		cugar::Vector3f U, V, W;
		camera_frame( s_renderer->get_camera(), s_renderer->get_aspect_ratio(), U, V, W );

		translate_group(
			s_renderer->get_device_mesh(),
			s_renderer->m_selected_group,
			s_renderer->m_mouse_depth * U * float(x - mx) / float(res.x) -
			s_renderer->m_mouse_depth * V * float(y - my) / float(res.y));

		s_renderer->m_dirty_geo = true;
		s_renderer->m_dirty		= true;

		// reset the mouse position
		s_renderer->m_mouse.x = x;
		s_renderer->m_mouse.y = y;
	}

	//fprintf(stderr, "mouse at %u, %u\n", x, y);

	glutPostRedisplay();
}

void start_glut_viewer(int argc, char** argv)
{
	GlutViewer renderer;
	s_renderer = &renderer;
	s_renderer->init(argc, argv);
}