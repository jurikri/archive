
function register(){
setBatchMode(true);
	
run("Bio-Formats Importer", "open=C:/mass_save/auto_turboreg_run/work2/D12.oir " + "autoscale color_mode=Default"); 
n=nSlices();
rawfile = getTitle();
run("Z Project...", "start=0 stop=n projection=[Average Intensity]"); 
avgimg = getTitle();

width = getWidth();
height = getHeight();



for(k=1; k <n; k++) { 
	
selectWindow(rawfile);
setSlice(k);
run("Duplicate...", "title=currentFrame");
	
run("TurboReg ", "-align " 
+ "-window currentFrame " 
+ "0 0 " + (width - 1) + " " + (height - 1) + " " // No cropping.
+ "-window "+ avgimg + " "// Target
+ "0 0 " + (width - 1) + " " + (height - 1) + " " // No cropping.
+ "-rigidBody "
+ "256 256 256 256 " //Landmark 1: translation
+ "256 80 256 80 " //Landmark 2: angle
+ "256 432 256 432 " //Landmark 3: angle
+ "-showOutput");

selectWindow("Output");
rename("output_img");
run("Duplicate...", "title=registered");
run("16-bit");

if(k==1)
{
run("Duplicate...", "title=mssave");
selectWindow("registered");
close();
}

else{
run("Concatenate...", "title=mssave" + " image1=mssave image2=registered image3=[-- None --] image4=[-- None --]");
}

selectWindow("output_img");
close();
selectWindow("currentFrame");
close();
showProgress(k+1,n);

}

setBatchMode(false);
}

register()