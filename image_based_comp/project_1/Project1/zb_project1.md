Zane Bates


Notes:
Segment just brain and not brain, in grey scale, and then use Point class to give data points to segment

Count how many voxels with non-zero intensity values -> in this guy above with brain and nonbrain segments

For problem 1, use page 23 of Manual -> gives you header information on how to import data, then use the data to get #Voxels = nonzero -> gives you dog brain volume

For problem 2:
Find Isosurface of molecule with app file, using the Add, Isocontour, and then open in Meshviewer as raw, take pic, then get the rawc file which is raw with a bit extra color, save pic, then write code to get potential on every point of isosurface/isocontour

it seems like the calculation with the trilinear only uses the matrices from the raw file, 
i think the trilinear interpolation is how we are calculating the rawc file

This one is a signed distance field file which is voxelized, and marching cubes is used to determine isosurface, which is a explicit triangle mesh -> why is the data point

all the values of the geometry are telling you the isosurface, where the value is between zero and one based on the parameters used to export the raw file from Volume Rover. 