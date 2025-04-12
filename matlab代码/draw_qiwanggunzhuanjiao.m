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

% ɾ���κ� NaN ֵ,ȷ�������Ǹɾ���
validIndices = ~(isnan(firstColumn) | isnan(secondColumn) | isnan(thirdColumn) | isnan(seventhColumn));
firstColumn = firstColumn(validIndices);
secondColumn = secondColumn(validIndices);
thirdColumn = thirdColumn(validIndices);
seventhColumn = seventhColumn(validIndices);

% �ӵ� 15062 �п�ʼ��ȡ����
startIndex = 15062;
firstColumn = firstColumn(startIndex:end);
seventhColumn = seventhColumn(startIndex:end);

% ���Ƶ�һ������
figure; % ����һ����ͼ�δ���
set(gcf, 'Color', 'w'); % ��������ͼ�δ��ڱ���Ϊ��ɫ
plot(seventhColumn, firstColumn, 'r-'); % ���Ƶ�һ������,��ɫʵ��
xlabel('ʱ��/t'); % Ϊ x ����ӱ�ǩ
ylabel('������Ͳ��ת��/��'); % Ϊ y ����ӱ�ǩ

% ע�͵���ɾ������һ���Բ���ʾͼ��
% legend('show'); % ���������У��Բ���ʾͼ��
