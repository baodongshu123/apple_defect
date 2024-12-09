
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


class FaceDataset(data.Dataset):
    def __init__(self,root_dir,label_dir):
        self.root_dir=root_dir
        self.label_dir=label_dir
        self.path=os.path.join(self.root_dir,self.label_dir)
        self.img_path=os.listdir(self.path)

    def __getitem__(self, index):
        img_name=self.img_path[index]
        img_item_path=os.path.join(self.root_dir,self.label_dir,img_name)
        img=Image.open(img_item_path).convert('RGB')
        img_array=np.array(img)
        img_array=cv2.resize(img_array,(224,224))
        img_array = img_array.reshape(224, 224, 3)/255.0
        tensortrans=transforms.ToTensor()
        img_tensor=tensortrans(img_array)
        label=self.label_dir
        label=int(label)
        return img_tensor,label

    def __len__(self):
        return len(self.img_path)

class SelfAttention(nn.Module):
    def __init__(self, in_channels):
        super(SelfAttention, self).__init__()
        self.query = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.key = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.value = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        batch_size, channels, height, width = x.size()
        query = self.query(x).view(batch_size, -1, height * width).permute(0, 2, 1)
        key = self.key(x).view(batch_size, -1, height * width)
        attention = F.softmax(torch.bmm(query, key), dim=2)
        value = self.value(x).view(batch_size, -1, height * width)
        out = torch.bmm(value, attention.permute(0, 2, 1))
        out = out.view(batch_size, channels, height, width)
        out = self.gamma * out + x
        return out




class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super(MultiHeadAttention, self).__init__()
        self.n_heads = n_heads
        self.d_model = d_model
        assert d_model % n_heads == 0,
        self.d_k = d_model // n_heads

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        batch_size, seq_len, height, width = x.size()

        Q = self.W_q(x.view(batch_size, seq_len, -1)).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        K = self.W_k(x.view(batch_size, seq_len, -1)).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        V = self.W_v(x.view(batch_size, seq_len, -1)).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.d_k ** 0.5)
        attn_weights = F.softmax(scores, dim=-1)

        attn_output = torch.matmul(attn_weights, V)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, height, width, -1)
        attn_output = self.W_o(attn_output.view(batch_size, seq_len, -1))
        out = self.gamma * attn_output + x
        return out


class improve_apple_one3(nn.Module):
    def __init__(self):
        super(improve_apple_one3, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=32),
            nn.ELU(),
            nn.MaxPool2d(kernel_size=2)
        )

        self.attention1 = SelfAttention(32)

        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=64),
            nn.ELU(),
            nn.MaxPool2d(kernel_size=2)
        )

        self.attention2 = SelfAttention(64)

        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=128),
            nn.ELU(),
            nn.MaxPool2d(kernel_size=2)
        )

        self.attention3 = SelfAttention(128)

        self.fc1 = nn.Linear(in_features=128 * 10 * 10, out_features=128)
        self.fc2 = nn.Linear(in_features=128, out_features=2)

    def forward(self, x):
        x = self.conv1(x)
        x = self.attention1(x)
        x = self.conv2(x)
        x = self.attention2(x)
        x = self.conv3(x)
        x = self.attention3(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = F.elu(x)
        x = self.fc2(x)
        return x

class improve_apple_one3_mult(nn.Module):

    def __init__(self, n_heads=4):
        super(improve_apple_one3_mult, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=32),
            nn.ELU(),
            nn.MaxPool2d(kernel_size=2)
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=64),
            nn.ELU(),
            nn.MaxPool2d(kernel_size=2)
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=128),
            nn.ELU(),
            nn.MaxPool2d(kernel_size=2)
        )

        self.multihead_attn = MultiHeadAttention(d_model=128, n_heads=n_heads)

        self.fc1 = nn.Linear(in_features=128*28*28*2, out_features=256)
        self.fc2 = nn.Linear(in_features=256, out_features=10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        print(x.shape)
        x = self.multihead_attn(x)
        print(x.shape)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = F.elu(x)
        x=self.fc2(x)
        return x







