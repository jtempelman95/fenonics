 
syms w1 w2 w3 w0 w4 w5 w6 w7
clc
n = 4;
series = 0
for j = 0:(n-1) 
for l = [0:(n-1-j)];
    i1 = j;
    i2 = n-1-j-l;
    i3 = l;
    [i1 i2 i3]
        
    if i1 == 0; a = w0; elseif i1 ==1; a = w1; elseif i1 ==2; a = w2; elseif i1 == 3; a = w3; elseif i1 == 4; a = w4; elseif i1 == 5; a = w5; elseif i1 == 6; a = w6; elseif i1 == 7; a = 7;end 
    if i2 == 0; b = w0; elseif i2 ==1; b = w1; elseif i2 ==2; b = w2; elseif i2 == 3; b = w3; elseif i2 == 4; b = w4; elseif i2 == 5; b = w5; elseif i2 == 6; b = w6; elseif i2 == 7; b = 7;end 
    if i3 == 0; c = w0; elseif i3 ==1; c = w1; elseif i3 ==2; c = w2; elseif i3 == 3; c = w3; elseif i3 == 4; c = w4; elseif i3 == 5; c = w5; elseif i3 == 6; c = w6; elseif i3 == 7; c = 7;end 
    
    series = series + a*b*c;
    
end
end
series
latex(series)


%%
syms u0 t u1 u2 u3 u01 u11 u13 u21 u23 u25
i = sqrt(-1)
x0 = u01*exp(i*t) +conj(u01)*exp(-i*t)
x1 = u11*exp(i*t) +conj(u11)*exp(-i*t)+u13*exp(3*i*t) +conj(u13)*exp(-3*i*t)
x2 = u21*exp(i*t) +conj(u21)*exp(-i*t)+u23*exp(3*i*t) +conj(u23)*exp(-3*i*t) +u25*exp(5*i*t) +conj(u25)*exp(-5*i*t)


soll=expand(x0^3)

mastersol = soll;


keep = 1
for k = -3:5
    if k~=keep
    sol = mastersol;
    sol = subs(sol,exp(-t*i*k),0);
    end
end

%%
clearvars
a        = 1.2;
r       = 2*[1,.5,.4,.5];
r                   = 30*[0.08575204, 0.05561746, 0.0635377 , 0.01611497];
r = [0.82658734, 0.30422512, 0.08036392, 0.61178111]
Nquad   = 4;
j =1 ;
xd =[];yd = [];
xd2 = []; yd2 = [];
for quad = 1:Nquad
    for k = 1:length(r)
        theta = 2*(k-1)*pi/length(r)/Nquad + 2*(quad-1)*pi/Nquad + 0/4*pi/Nquad ;

        xd(j) =0/2 +  r(k)/2  * cos(theta);
        yd(j) =0/2 +  r(k)/2  * sin(theta);


        xd2(j) =0/2 +  r(k)/2  * cos(theta+pi/2);
        yd2(j) =0/2 +  r(k)/2  * sin(theta+pi/2);
        j = j + 1;
    end
end

figure(1)
clf
hold on
plot(xd(1:length(r)),yd(1:length(r)),'o')
plot(xd2(1:length(r)),yd2(1:length(r)),'o')
plot(xd,yd,'-.')
xy = [xd xd(1);yd yd(1)];
fnplt(cscvn(xy),'r',1)


xy2 = [xd2 xd2(1);yd2 yd2(1)];
fnplt(cscvn(xy2),'c--',1)

points = .95*[0 -1 0 1;1 0 -1 0];
points = xy;
sp = spmak(1:37,[points points]);
plot(points(1,:),points(2,:),'x'), hold on 
fnplt(sp,[2,18],'b'), 


points = xy2;
sp = spmak(1:37,[points points]);
plot(points(1,:),points(2,:),'x'), hold on 
fnplt(sp,[2,18],'m--'), axis equal square, hold off

grid on
axis equal

% figure(2);clf
% npts = 10
% xy = [randn(1,npts); randn(1,npts)];
% plot(xy(1,:),xy(2,:),'ro','LineWidth',2);
% text(xy(1,:), xy(2,:),[repmat('  ',npts,1), num2str((1:npts)')])
% fnplt(cscvn(xy),'r',2)
