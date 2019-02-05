REM for /r %%i in (*) do python codelinks.py %%i

python codelinks.py ../docs/html/_b_p_t_lib_page.html
python codelinks.py ../docs/html/_hello_renderer_page.html
python codelinks.py ../docs/html/_p_s_f_p_t_page.html

del ..\docs\html\*.orig
