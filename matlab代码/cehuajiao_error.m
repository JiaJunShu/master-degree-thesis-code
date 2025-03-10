% ��ȡ��һ��txt�ļ��е�����
data1 = importdata('mtkl_method1_cehuajiao_v220_4320,0.txt');

% ��ȡÿһ��
x_coords1 = data1(:, 1);  % ��һ������
y_coords1 = data1(:, 2);  % �ڶ�������
z_coords1 = data1(:, 3);  % ����������

% ������� (4320, 0) �ľ���
reference_point = [4320, 0];  % �ο��� (4320, 0)
distances1 = sqrt((x_coords1 - reference_point(1)).^2 + (y_coords1 - reference_point(2)).^2);

% �� z_coords1 ��С�������򣬲�ͬ������ distances1
[z_coords_sorted1, sort_idx1] = sort(z_coords1);
distances_sorted1 = distances1(sort_idx1);

% ʹ�� polyfit ���ж���ʽ��ϣ����������ϣ�
p1 = polyfit(z_coords_sorted1, distances_sorted1, 5);  % 7 ��ʾ�ߴ����

% ����������ߵ� x ֵ
z_fit1 = linspace(min(z_coords_sorted1), max(z_coords_sorted1), 100);

% ����������ߵ� y ֵ
distances_fit1 = polyval(p1, z_fit1);

% ��ȡ�ڶ���txt�ļ��е�����
data2 = importdata('mtkl_method2_cehuajiao_v220_4320,0.txt');  % �滻Ϊ�ڶ����ļ���ʵ��·��

% ��ȡÿһ��
x_coords2 = data2(:, 1);  % ��һ������
y_coords2 = data2(:, 2);  % �ڶ�������
z_coords2 = data2(:, 3);  % ����������

% ������� (4320, 0) �ľ���
distances2 = sqrt((x_coords2 - reference_point(1)).^2 + (y_coords2 - reference_point(2)).^2);

% �� z_coords2 ��С�������򣬲�ͬ������ distances2
[z_coords_sorted2, sort_idx2] = sort(z_coords2);
distances_sorted2 = distances2(sort_idx2);

% ʹ�� polyfit ���ж���ʽ��ϣ����������ϣ�
p2 = polyfit(z_coords_sorted2, distances_sorted2, 5);  % 7 ��ʾ�ߴ����

% ����������ߵ� x ֵ
z_fit2 = linspace(min(z_coords_sorted2), max(z_coords_sorted2), 100);

% ����������ߵ� y ֵ
distances_fit2 = polyval(p2, z_fit2);


% �����������
figure;
%plot(z_coords_sorted1, distances_sorted1, 'o', 'DisplayName', '�ļ�1ԭʼ����');  % �ļ�1ԭʼ���ݵ�
hold on;
set(gcf, 'Color', 'w');  % ���ñ�����ɫΪ��ɫ
plot(z_fit1, distances_fit1, '-', 'DisplayName', '����һ');  % �ļ�1�������
%plot(z_coords_sorted2, distances_sorted2, 's', 'DisplayName', '�ļ�2ԭʼ����');  % �ļ�2ԭʼ���ݵ�
plot(z_fit2, distances_fit2, '-', 'DisplayName', '������');  % �ļ�2�������
xlabel('��ʼ���/��');
ylabel('���/m');
%title('�����ļ����������');
legend;
%grid on;
hold off;
