import os
import shutil
import sys
import zipfile


pkgdir = "/tmp/sls-py-req"

sys.path.append(pkgdir)

if not os.path.exists(pkgdir):
    tempdir = "/tmp/_temp-sls-py-req"
    if os.path.exists(tempdir):
        shutil.rmtree(tempdir)

    default_layer_root = "/opt"
    lambda_root = os.getcwd() if os.environ.get("IS_LOCAL") == "true" else default_layer_root
    zip_requirements = os.path.join(lambda_root, ".requirements.zip")

    zipfile.ZipFile(zip_requirements, "r").extractall(tempdir)
    os.rename(tempdir, pkgdir)  # Atomic
