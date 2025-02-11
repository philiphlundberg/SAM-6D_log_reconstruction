import open3d as o3d

ply_path = "/home/philiph/Documents/PhiliphExjobb/automatic_scene_reconstruction/SAM-6D/SAM-6D/Data/Example/cyl.ply"
mesh = o3d.io.read_triangle_mesh(ply_path)


# Write as ASCII by specifying write_ascii=True
# Specify the output file name to be the same as the input name but add _ascii
output_name = ply_path.replace(".ply", "_ascii.ply")
o3d.io.write_triangle_mesh(output_name, mesh, write_ascii=True)