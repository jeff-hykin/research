require 'fileutils'
all_files = `locate pastie`
for each in all_files.split("\n")
    if each =~ /.*\bpastie\b.*/
        # FileUtils.delete(each)
        puts each
    end
end

/etc/apt/sources.list.d/hel-sheep-ubuntu-pastie-disco.list
/etc/xdg/autostart/pastie-startup.desktop
/home/jeff-hykin/anaconda3/lib/python3.7/site-packages/pygments/styles/pastie.py
/home/jeff-hykin/anaconda3/lib/python3.7/site-packages/pygments/styles/__pycache__/pastie.cpython-37.pyc
/home/jeff-hykin/anaconda3/pkgs/pygments-2.3.1-py37_0/lib/python3.7/site-packages/pygments/styles/pastie.py
/home/jeff-hykin/anaconda3/pkgs/pygments-2.3.1-py37_0/lib/python3.7/site-packages/pygments/styles/__pycache__/pastie.cpython-37.pyc
/home/jeff-hykin/anaconda3/pkgs/pygments-2.4.0-py_0/site-packages/pygments/styles/pastie.py
/usr/bin/pastie
/usr/share/applications/pastie.desktop
/usr/share/doc/pastie
/usr/share/doc/pastie/changelog.gz
/usr/share/doc/pastie/copyright
/usr/share/gconf/schemas/pastie.schemas
/usr/share/locale/cs/LC_MESSAGES/pastie.mo
/usr/share/locale/de/LC_MESSAGES/pastie.mo
/usr/share/locale/es/LC_MESSAGES/pastie.mo
/usr/share/locale/fi/LC_MESSAGES/pastie.mo
/usr/share/locale/fr/LC_MESSAGES/pastie.mo
/usr/share/locale/he/LC_MESSAGES/pastie.mo
/usr/share/locale/it/LC_MESSAGES/pastie.mo
/usr/share/locale/ja/LC_MESSAGES/pastie.mo
/usr/share/locale/pt_BR/LC_MESSAGES/pastie.mo
/usr/share/locale/ru/LC_MESSAGES/pastie.mo
/usr/share/locale/uk/LC_MESSAGES/pastie.mo
/usr/share/pyshared/pastie-0.6.1-py2.6.egg-info
/usr/share/python-support/pastie.public
/var/crash/pastie.0.crash
/var/lib/dpkg/info/pastie.conffiles
/var/lib/dpkg/info/pastie.list
/var/lib/dpkg/info/pastie.md5sums
/var/lib/dpkg/info/pastie.postinst
/var/lib/dpkg/info/pastie.prerm