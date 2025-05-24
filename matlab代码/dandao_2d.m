% �����ļ���Ϊ "output.txt"
filename = 'output.txt';

% ʹ���Ʊ����Ϊ�ָ�����ȡ�ļ�
opts = detectImportOptions(filename, 'Delimiter', '\t');

% ȷ���ļ����㹻���У���ָ����ȡ���У����ģ�Var4���͵��壨Var5����
opts.SelectedVariableNames = opts.VariableNames([4, 6]);

% �����ݶ�����
data = readtable(filename, opts);

% �������к͵�����ת��Ϊ����
x = table2array(data(:, 1)); % �����У������꣩
y = table2array(data(:, 2)); % �����У������꣩

% ��������ͼ
figure;
set(gcf, 'Color', 'w'); % ��������ͼ�δ��ڱ���Ϊ��ɫ
plot(x, y, '-', 'LineWidth', 1, 'Color', 'k'); % ����������
hold on; % ���ֵ�ǰͼ��

% ���������յ�
plot(x(1), y(1), 'ro', 'MarkerSize', 8, 'LineWidth', 2); % ��㣬��ɫԲȦ
plot(x(end), y(end), 'bs', 'MarkerSize', 8, 'LineWidth', 2); % �յ㣬��ɫ����

% ֱ���������յ�λ���������ע��
text(x(1), y(1), '���', 'FontSize', 10, 'Color', 'black', ...
    'VerticalAlignment', 'bottom', 'HorizontalAlignment', 'right'); % ����ǩ
text(x(end), y(end), '�յ�', 'FontSize', 10, 'Color', 'black', ...
    'VerticalAlignment', 'top', 'HorizontalAlignment', 'left'); % �յ��ǩ

% ����������ǩ������
xlabel('X/m');
ylabel('Z/m');

%grid on; % ��ʾ����
