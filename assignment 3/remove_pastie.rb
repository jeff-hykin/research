require 'fileutils'
all_files = Dir.glob('/**/*pastie')
for each in all_files
    if each =~ /\bpastie\b/
        # FileUtils.delete(each)
    end
end