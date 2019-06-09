require 'fileutils'
Dir.chdir __dir__
all_training = Dir.glob("cats_and_dogs_train.nosync/*")
validation_set = "cats_and_dogs_validate.nosync"
FileUtils.mkdir(validation_set)
# move the last 1000 images from training to validation
for each in all_training[-999..-1]
    FileUtils.move(each, validation_set)
end