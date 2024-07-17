datapath = 'E:\Code\simplified_l5pn\results\stochastic';
tau_all = [1,3,5,10,20,50,80,100,200,500];
data = dir(fullfile(datapath, '*.mat'));
mean_b1 = zeros(length(tau_all), 8);
std_b1 = zeros(length(tau_all), 8);
mean_b2 = zeros(length(tau_all), 8);
std_b2 = zeros(length(tau_all), 8);
p_b1 = zeros(length(tau_all), 8);
p_b2 = zeros(length(tau_all), 8);
for j = 1:length(data)
    load (fullfile(data(j).folder, data(j).name))
    idx = find(tau_all == tau);
    if length(idx)==1
        mean_b1(idx,:) = mean(burstiness(:,:,1),2)';
        mean_b2(idx,:) = mean(burstiness(:,:,2),2)';
        std_b1(idx,:) = std(burstiness(:,:,1),1,2)';
        std_b2(idx,:) = std(burstiness(:,:,2),1,2)';
        for ii = 1:length(amp)
            [p_b1(idx,ii),h] = signrank(burstiness(1,:,1), burstiness(ii,:,1));
            [p_b2(idx,ii),h] = signrank(burstiness(1,:,2), burstiness(ii,:,2));
        end
    end
end
colors = [[128,128,128];[119,177,204];[61,139,191];[6,50,99]];
colors = colors/256;
baseline = mean([mean_b1(:,1);mean_b2(:,1)]);
figure
plot(amp, mean_b1(find(tau_all==10),:),'Color', colors(1,:), 'Linewidth', 2)
hold on
plot(amp, mean_b1(find(tau_all==10),:)+std_b1(find(tau_all==10),:)/sqrt(10),'Color', colors(1,:), 'Linewidth', 0.5)
plot(amp, mean_b1(find(tau_all==10),:)-std_b1(find(tau_all==10),:)/sqrt(10),'Color', colors(1,:), 'Linewidth', 0.5)

plot(amp, mean_b2(find(tau_all==10),:),'Color', colors(3,:), 'Linewidth', 2)
plot(amp, mean_b2(find(tau_all==10),:)+std_b2(find(tau_all==10),:)/sqrt(10),'Color', colors(3,:), 'Linewidth', 0.5)
plot(amp, mean_b2(find(tau_all==10),:)-std_b2(find(tau_all==10),:)/sqrt(10),'Color', colors(3,:), 'Linewidth', 0.5)
hold on
plot(amp, baseline*ones(1, length(amp)))

figure
plot(tau_all, mean_b1(:,find(amp==0.14)),'Color', colors(1,:), 'Linewidth', 2)
hold on
plot(tau_all, mean_b1(:,find(amp==0.14))+std_b1(:,find(amp==0.14))/sqrt(10),'Color', colors(1,:), 'Linewidth', 0.5)
plot(tau_all, mean_b1(:,find(amp==0.14))-std_b1(:,find(amp==0.14))/sqrt(10),'Color', colors(1,:), 'Linewidth', 0.5)

plot(tau_all, mean_b2(:,find(amp==0.14)),'Color', colors(3,:), 'Linewidth', 2)
plot(tau_all, mean_b2(:,find(amp==0.14))+std_b2(:,find(amp==0.14))/sqrt(10),'Color', colors(3,:), 'Linewidth', 0.5)
plot(tau_all, mean_b2(:,find(amp==0.14))-std_b2(:,find(amp==0.14))/sqrt(10),'Color', colors(3,:), 'Linewidth', 0.5)
set(gca,'xscale','log')
hold on
plot(tau_all, baseline*ones(1, length(tau_all)))

cmap = custom_cmap('darkblue');
figure, imagesc(mean_b1),colormap(flip(cmap,1)), colorbar(),caxis([0.1,0.5])
figure, imagesc(mean_b2),colormap(flip(cmap,1)), colorbar(),caxis([0.1,0.5])

[X,Y] = meshgrid(amp*1e3,tau_all);
figure;h = gca;s = surf(X,Y,mean_b1);colormap(flip(cmap,1));set(h,'yscale','log'),view(2),ylim([0,500]),colorbar(),caxis([0.1,0.5])
figure;h = gca;s = surf(X,Y,mean_b2);colormap(flip(cmap,1));set(h,'yscale','log'),view(2),ylim([0,500]),colorbar(),caxis([0.1,0.5])