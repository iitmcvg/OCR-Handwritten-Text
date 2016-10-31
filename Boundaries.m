%tracing boundary regions

Im = imread('C:\Advaith\CVG Project\text4.jpg');
Img=im2bw(Im,graythresh(Im));
Img=imcomplement(Img);
[B,L] = bwboundaries(Img,'noholes');


Black=im2bw(Im,1);    
imshow(Black);
hold on;
colors=['b' 'g' 'r' 'c' 'm' 'y'];
for k=1:length(B),
  boundary = B{k};
  cidx = mod(k,length(colors))+1;
  ImagePlusPlot= plot(boundary(:,2), boundary(:,1),...
       colors(cidx),'LineWidth',2);
end

F = getframe(gcf);
[Im, Map] = frame2im(F);

%regionprops

measurements=regionprops(Im,'BoundingBox');
totalNumberOfBlobs = length(measurements);
imshow(Im);
hold on;

for blobNumber = 1:totalNumberOfBlobs
    bb = measurements(blobNumber).BoundingBox;
    %bb=[bb(1:2) bb(4:5)];
    rectangle('Position',bb,'EdgeColor','r','LineWidth',2);
end