gt=[gt_r,gt_t];
pred=[pred_r,pred_t];
p=[0;0;0;1];
g=[0;0;0;1];
figure(1)
for i=1:100
    pred_point=[reshape(pred(i,1:9),[3,3])',reshape(pred(i,10:12),[3,1]);[0,0,0,1]]*p;
    p=pred_point;
    pred(i,1:9)
    plot3(p(1),p(2),p(3),'g*-')
    hold on
end
figure(1);
for i=1:100
    g_point=[reshape(gt(i,1:9),[3,3])',reshape(gt(i,10:12),[3,1]);[0,0,0,1]]*g;
    g=g_point
    plot3(g(1),g(2),g(3),'*b-')
    hold on
end