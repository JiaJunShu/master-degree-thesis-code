% ��ȡ�ı��ļ�
data = load('mtkl_method4_v280_6240,0.txt'); % ʹ�� load ��ȡ����

% ��ȡ��
coords_x = data(:, 1); % ��һ������ (ʵ������)
coords_y = data(:, 2); % �ڶ������� (ʵ������)
x_coords = data(:, 3);  % ��������Ϊ����ϵ��X����
y_coords = data(:, 4);  % ��������Ϊ����ϵ��Y����

% ����ο���
reference_point = [6240,0];

% ����ÿ������Ͳο���ľ���
distances = sqrt((coords_x - reference_point(1)).^2 + (coords_y - reference_point(2)).^2);

% ׼����ͼ
figure;
set(gcf, 'Color', 'w');  % ���ñ�����ɫΪ��ɫ
% ��������
[X, Y] = meshgrid(linspace(min(x_coords), max(x_coords), 100), ...
                 linspace(min(y_coords), max(y_coords), 100));

% ����һ�β�ֵ���� Z ֵ
Z = griddata(x_coords, y_coords, distances, X, Y, 'linear'); % ʹ�����Բ�ֵ

% ��������
surf(X, Y, Z, 'EdgeColor', 'none'); % �������棬��ȥ���߽���
colorbar; % ��ʾ��ɫ��
xlabel('����ƫ��/��');
ylabel('��ʼ��Ͳ��ת��/��');
zlabel('���/m');
%title('Surface Plot of Distances (Linear Interpolation)');
view(30, 30); % �����ӽ�
grid on;

% ��ӵȸ���
hold on;  % ���ֵ�ǰͼ��
contour3(X, Y, Z, 5, 'LineColor', 'k'); % ���Ƶȸ��ߣ�20���ȸ���
hold off; % �ͷ�ͼ��
