% ��ȡ�����Էָ��Ʊ���Ϳո�
data = readtable('mtkl_method1_wind_zong_v220_4320,0.txt', 'Delimiter', ' ', 'ReadVariableNames', false);
data2 = readtable('mtkl_method2_wind_zong_v220_4320,0.txt', 'Delimiter', ' ', 'ReadVariableNames', false);

% ��ȡÿ������
y = data{:, 1}; % ��һ���������
x = data{:, 2}; % ��һ��ĺ�����
y2 = data2{:, 1}; % �ڶ����������
x2 = data2{:, 2}; % �ڶ���ĺ�����

% ����ɢ��ͼ
scatter(x, y, 'o', 'filled'); % ��һ��������Բ���ʾ
hold on; % ���ֵ�ǰͼ��
scatter(x2, y2, 'x'); % �ڶ��������������α�ʾ



% ���� (4350, 4300) �ú�ɫ���
highlight_x = 0; % ������ x ����
highlight_y = 4320; % ������ y ����
scatter(highlight_x, highlight_y, 100, 'red', 'filled'); % ������һЩ

% ��ӱ�ע
%text(highlight_x, highlight_y, '  (6275, 0)', 'Color', 'red', 'FontSize', 10); 

set(gcf, 'Color', 'w');  % ���ñ�����ɫΪ��ɫ


% �� (4300, 60) ΪԲ�Ļ�Բ
circle_highlight_x = 0; % Բ�� x ����
circle_highlight_y = 4320; % Բ�� y ����
radius = 10; % �뾶
theta = linspace(0, 2*pi, 100); % Բ�ܽǶ�
circle_x = circle_highlight_x + radius * cos(theta); % Բ�� x ����
circle_y = circle_highlight_y + radius * sin(theta); % Բ�� y ����
plot(circle_x, circle_y, 'r-', 'LineWidth', 1.5); % ����Բ����ɫ����

% ��ӱ�����������ǩ
xlabel('Y/m');
ylabel('X/m');


% ���ͼ��
legend('����һ', '������','Ŀ���', 'Location', 'northeast'); 

% �����᷶Χ
xlim([-12, 12]); % �µ� x ���귶Χ
ylim([4308,4332]); % �µ� y ���귶Χ

hold off; % ������ǰͼ�εı���
