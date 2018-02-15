import struct

def to_csv(name, maxdata):
  # open label file and image file
  lbl_f = open("./mnist/"+name+"-labels-idx1-ubyte", "rb")
  img_f = open("./mnist/"+name+"-images-idx3-ubyte", "rb")
  csv_f = open("./mnist/"+name+".csv", "w", encoding="utf-8")

  # read header information
  mag, lbl_count = struct.unpack(">II", lbl_f.read(8))
  mag, img_count = struct.unpack(">II", img_f.read(8))
  rows, cols = struct.unpack(">II", img_f.read(8))
  pixels = rows * cols

  # read image data and save it as csv files
  res = []
  for idx in range(lbl_count):
    if idx > maxdata: break
    label = struct.unpack("B", lbl_f.read(1))[0]
    bdata = img_f.read(pixels)
    sdata = list(map(lambda n: str(n), bdata))
    csv_f.write(str(label)+",")
    csv_f.write(",".join(sdata)+"\r\n")

    # test if it is saved well: ten data
    if idx < 10:
      s = "P2 28 28 255\n"
      s += " ".join(sdata)
      iname = "./mnist/{0}-{1}-{2}.pgm".format(name, idx, label)
      with open(iname, "w", encoding="utf-8") as f:
        f.write(s)
  
  csv_f.close()
  lbl_f.close()
  img_f.close()

to_csv("train", 1000)
to_csv("t10k", 500)
