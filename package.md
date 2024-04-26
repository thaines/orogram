# Package

To package ready for deployment:

```
sudo -H python3 -m pip install --upgrade build
python3 -m build
```

This creates
```
dist/Orogram-0.6.1.tar.gz
dist/Orogram-0.6.1-cp36-cp36m-linux_x86_64.whl
```



# Repair and validate

```
sudo -H python3 -m pip install --upgrade auditwheel==5.1.2
sudo apt install patchelf

auditwheel show dist/Orogram-0.6.1-cp36-cp36m-linux_x86_64.whl

auditwheel repair --plat manylinux_2_17_x86_64 dist/Orogram-0.6.1-cp36-cp36m-linux_x86_64.whl
auditwheel show wheelhouse/Orogram-0.6.1-cp36-cp36m-manylinux_2_17_x86_64.manylinux2014_x86_64.whl
```



# Deploy

See [https://pypi.org/manage/account/token/](https://pypi.org/manage/account/token/) and get as far as a `.pypirc` file.

Then
```
sudo -H python3 -m pip install --upgrade twine
python3 -m twine upload --repository pypi dist/Orogram-0.6.1.tar.gz
python3 -m twine upload --repository pypi wheelhouse/Orogram-0.6.1-cp36-cp36m-manylinux_2_17_x86_64.manylinux2014_x86_64.whl
```

