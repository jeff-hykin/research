require 'fileutils'
files = [
    "/etc/apt/sources.list.d/hel-sheep-ubuntu-pastie-disco.list",
    "/etc/xdg/autostart/pastie-startup.desktop",
    "/usr/bin/pastie",
    "/usr/share/applications/pastie.desktop",
    "/usr/share/doc/pastie",
    "/usr/share/doc/pastie/changelog.gz",
    "/usr/share/doc/pastie/copyright",
    "/usr/share/gconf/schemas/pastie.schemas",
    "/var/crash/pastie.0.crash",
    "/var/lib/dpkg/info/pastie.conffiles",
    "/var/lib/dpkg/info/pastie.list",
    "/var/lib/dpkg/info/pastie.md5sums",
    "/var/lib/dpkg/info/pastie.postinst",
    "/var/lib/dpkg/info/pastie.prerm"
]
for each in files
    system "sudo rm '#{each}'"
end