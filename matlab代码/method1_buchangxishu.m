% ����������������
x = [-174,-165,-153,-133,-120,-109,-100,-90,-86,-70,-50,-30,-10,20,30,60,80,100,124,140,168,170];
y = [-163,-136,-132,-110,-95,-85,-82,-60,-50,-25,-17,-5,0,1.3,7,19,30,50,88,110,146,158];

% �����ֵ
ratios = y ./ x;

% ����ͼ��
figure;
set(gcf, 'Color', 'w'); % ��������ͼ�δ��ڱ���Ϊ��ɫ
% ʹ�ò�ֵ����ƽ��������
xq = linspace(min(x), max(x),100); % ����һ��ϸ�ֵ�x�������ڲ�ֵ
ratios_smooth = interp1(x, ratios, xq, 'spline'); % ʹ��������ֵ����ƽ��

% ����ԭʼ���ݵ�
%plot(x, ratios, 'o', 'DisplayName', 'Data Points');
hold on;

% ����ƽ������
plot(xq, ratios_smooth, 'r-', 'DisplayName', 'Smooth Curve');

% ��ӱ�����������ǩ
%title('Smooth Connection of Ratios');
xlabel('����ǰ�ĽǶ�');
ylabel('K');

% ���ͼ��
legend show;

% ��ʾ����
%grid on;
hold off;
