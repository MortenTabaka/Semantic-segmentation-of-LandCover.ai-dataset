# https://github.com/wkentaro/gdown

import os
import gdown

                
abs_path = os.path.abspath('')

DATA_DIR = abs_path + '/data/raw/'

output = DATA_DIR + 'landcover.zip'

id = "1xILwUliGlWw5hmMt4E4GdDn5yV3ln8_G"
gdown.download(id=id, output=output, quiet=False)
