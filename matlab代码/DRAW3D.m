% ��ȡ�ļ�����ȡ����
data = load('output.txt');

% ��ȡ x, y �� z ����
x = data(:, 4);
y = data(:, 5);
z = data(:, 6);

% ����3Dͼ��
figure;
set(gcf, 'Color', 'w'); % ��������ͼ�δ��ڱ���Ϊ��ɫ
plot3(x, y, z, '-o', 'LineWidth', 0.1);

% ��ע���
hold on;
plot3(x(1), y(1), z(1), 'ro', 'MarkerSize', 10, 'MarkerFaceColor', 'r');
text(x(1), y(1), z(1), ' ���', 'VerticalAlignment', 'bottom', 'HorizontalAlignment', 'right');

% ��ע�յ�
plot3(x(end), y(end), z(end), 'bo', 'MarkerSize', 10, 'MarkerFaceColor', 'b');
text(x(end), y(end), z(end), ' �յ�', 'VerticalAlignment', 'bottom', 'HorizontalAlignment', 'right');

% ��ʾ����
grid on;
xlabel('X/m');
ylabel('Y/m');
zlabel('Z/m');

grid off; % �ر�����
hold off;
