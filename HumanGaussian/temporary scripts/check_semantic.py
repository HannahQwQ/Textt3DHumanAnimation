from plyfile import PlyData
import sys

path = "content/save_ply/semantic-T-pose-boy.ply"
ply = PlyData.read(path)
el = ply.elements[0]
print("Element name:", el.name)
print("Num vertices:", el.count)
print("Property names:", [p.name for p in el.properties])
# also show dtype names
if hasattr(el, "data"):
    print("Data dtype names:", el.data.dtype.names)
# show first 10 entries (x,y,z and any 'semantic'-like fields)
first = el.data[:10]
print("First row keys available and example values:")
for k in first.dtype.names:
    print(f" {k}: {first[k][:5]}")
