Env dependency
-python
-opencv2
-numpy

*advise 
-use conda to create env

*run
-quick run(use default arguments)
--- python main.py 

-python main.py --help
-python main.py [arguments]

*arguments
--input_dir	/* input path */
--output_dir	/* output path */
--tone_map	/* Tone mapping method */
--n		/* Number of sampled points for Debevec method */
--d		/* Number of depth for image alignment */
--s		/* Image Scaling ratio */
--a		/* Alpha for photographic tonemapping */