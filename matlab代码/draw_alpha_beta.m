 % �����ļ���
filename = 'output.txt'; % �����ʵ���ļ����޸�

% ����1: ʹ�� readtable ��ȡ�ļ�
dataTable = readtable(filename);

% ��ȡǰ��������
firstColumn = dataTable{:, 1}; % ��һ��
secondColumn = dataTable{:, 2}; % �ڶ���
thirdColumn = dataTable{:, 3}; % ������
seventhColumn = dataTable{:, 7}; % ������

% ��鲢ת��������Ϊ��Ԫ�����
columnsToCheck = {firstColumn, secondColumn, thirdColumn, seventhColumn};
for i = 1:length(columnsToCheck)
    if iscell(columnsToCheck{i})
        columnsToCheck{i} = str2double(columnsToCheck{i});
    else
        columnsToCheck{i} = double(columnsToCheck{i});
    end
end

% ������ת������
[firstColumn, secondColumn, thirdColumn, seventhColumn] = deal(columnsToCheck{:});

% ɾ���κ� NaN ֵ��ȷ�������Ǹɾ���
validIndices = ~(isnan(firstColumn) | isnan(secondColumn) | isnan(thirdColumn) | isnan(seventhColumn));
firstColumn = firstColumn(validIndices);
secondColumn = secondColumn(validIndices);
thirdColumn = thirdColumn(validIndices);
seventhColumn = seventhColumn(validIndices);

% ���Ƶ�һ������
figure; % ����һ����ͼ�δ���
set(gcf, 'Color', 'w'); % ��������ͼ�δ��ڱ���Ϊ��ɫ
plot(seventhColumn, firstColumn, 'r-', 'DisplayName', '��һ������'); % ���Ƶ�һ�����ݣ���ɫʵ��
xlabel('ʱ��/t'); % Ϊ x ����ӱ�ǩ
ylabel('��Ͳ��ת��/��'); % Ϊ y ����ӱ�ǩ
%title('������Ϊ������ ��һ������ͼ'); % ��ӱ���
%grid on; % �������
legend('show'); % ��ʾͼ��

% ���Ƶڶ�������
figure; % ������һ����ͼ�δ���
set(gcf, 'Color', 'w'); % ��������ͼ�δ��ڱ���Ϊ��ɫ
plot(seventhColumn, secondColumn, 'b-', 'DisplayName', '�ڶ�������'); % ���Ƶڶ������ݣ���ɫʵ��
xlabel('����������'); % Ϊ x ����ӱ�ǩ
ylabel('ֵ'); % Ϊ y ����ӱ�ǩ
title('������Ϊ������ �ڶ�������ͼ'); % ��ӱ���
grid on; % �������
legend('show'); % ��ʾͼ��

% ���Ƶ���������
figure; % ������һ����ͼ�δ���
set(gcf, 'Color', 'w'); % ��������ͼ�δ��ڱ���Ϊ��ɫ
plot(seventhColumn, thirdColumn, 'g-', 'DisplayName', '����������'); % ���Ƶ��������ݣ���ɫʵ��
xlabel('����������'); % Ϊ x ����ӱ�ǩ
ylabel('ֵ'); % Ϊ y ����ӱ�ǩ
title('������Ϊ������ ����������ͼ'); % ��ӱ���
grid on; % �������
legend('show'); % ��ʾͼ��
