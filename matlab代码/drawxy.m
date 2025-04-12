% ��ȡ�����Էָ��Ʊ���Ϳո�
data = readtable('round2_luodianfanwei_v220.txt', 'Delimiter', {' ', '\t'}, 'MultipleDelimsAsOne', true, 'ReadVariableNames', false);


% ��ȡÿ������
y = data{:, 1}; % ��һ����Ϊ������
x = data{:, 2}; % �ڶ�����Ϊ������



% ����ɢ��ͼ
figure;

highlight_x=0;
highlight_y=4320;
scatter(highlight_x, highlight_y, 100, 'red', 'filled');
text(highlight_x, highlight_y, '  (4320,0)', 'Color', 'red', 'FontSize', 10); 
set(gcf, 'Color', 'w'); % ��������ͼ�δ��ڱ���Ϊ��ɫ
%scatter(x, y, 'filled'); % ����ɢ��ͼ��'filled'��ʾ�����
hold on; % ���ֵ�ǰͼ��
plot(x, y, 'k-'); % ��������
plot([x(1), x(end)], [y(1), y(end)],  'k-'); % �������һ����͵�һ����
xlabel('��ƫy/m');
ylabel('���x/m');
%title('��㷶Χ');
%grid on; % ������ʾ����
text(0, 4445, '0��', 'HorizontalAlignment', 'center', 'VerticalAlignment', 'bottom');
text(99, 4312, '90��', 'HorizontalAlignment', 'center', 'VerticalAlignment', 'bottom');
text(-109, 4313, '-90��', 'HorizontalAlignment', 'center', 'VerticalAlignment', 'bottom');
text(-2, 4200, '180��', 'HorizontalAlignment', 'center', 'VerticalAlignment', 'bottom');



% ��ͼ�������������"��"
text(80, 4400, '��', 'HorizontalAlignment', 'center', 'VerticalAlignment', 'bottom', 'FontSize', 12, 'FontWeight', 'bold');

% ��ͼ�������������"��"
text(80, 4250, '��', 'HorizontalAlignment', 'center', 'VerticalAlignment', 'bottom', 'FontSize', 12, 'FontWeight', 'bold');

% ��ͼ�������������"��"
text(-90, 4250, '��', 'HorizontalAlignment', 'center', 'VerticalAlignment', 'bottom', 'FontSize', 12, 'FontWeight', 'bold');

% ��ͼ�������������"��"
text(-80, 4400, '��', 'HorizontalAlignment', 'center', 'VerticalAlignment', 'bottom', 'FontSize', 12, 'FontWeight', 'bold');

hold off; % �ͷŵ�ǰͼ��