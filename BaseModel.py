# Movielens dataset
import torch
from torch.autograd import Variable
from torch.nn import functional as F

class UserEmbeddingML(torch.nn.Module):
    def __init__(self, config):
        super(UserEmbeddingML, self).__init__()

        self.num_user = config['num_user']
        self.num_gender = config['num_gender']
        self.num_age = config['num_age']
        self.num_occupation = config['num_occupation']
        self.num_zipcode = config['num_zipcode']

        self.embedding_dim = config['embedding_dim']

        self.embedding_userId = torch.nn.Embedding(
            num_embeddings=self.num_user,
            embedding_dim=self.embedding_dim
        )  # torch.nn.Embedding()
        self.embedding_gender = torch.nn.Embedding(
            num_embeddings=self.num_gender,
            embedding_dim=self.embedding_dim
        ) # torch.nn.Embedding()
        self.embedding_age = torch.nn.Embedding(
            num_embeddings=self.num_age,
            embedding_dim=self.embedding_dim
        )
        self.embedding_occupation = torch.nn.Embedding(
            num_embeddings=self.num_occupation,
            embedding_dim=self.embedding_dim
        )
        self.embedding_area = torch.nn.Embedding(
            num_embeddings=self.num_zipcode,
            embedding_dim=self.embedding_dim
        )

    def forward(self, user_fea):
        """
        :param user_fea: [userId(0),gender(1),age(2),occupation(3),area(4)]
        :return:
        """
        userId_idx = Variable(user_fea[:, 0], requires_grad=False)
        gender_idx = Variable(user_fea[:, 1], requires_grad=False)
        age_idx = Variable(user_fea[:, 2], requires_grad=False)
        occupation_idx = Variable(user_fea[:, 3], requires_grad=False)
        area_idx = Variable(user_fea[:, 4], requires_grad=False)

        userId_emb = self.embedding_userId(userId_idx)
        gender_emb = self.embedding_gender(gender_idx)
        age_emb = self.embedding_age(age_idx)
        occupation_emb = self.embedding_occupation(occupation_idx)
        area_emb = self.embedding_area(area_idx)
        return torch.cat((userId_emb, gender_emb, age_emb, occupation_emb, area_emb), 1)   # (samples, 5*32)  torch.cat() concatenation

class ItemEmbeddingML(torch.nn.Module):
    def __init__(self, config):
        super(ItemEmbeddingML, self).__init__()
        self.num_item = config['num_item']
        self.num_rate = config['num_rate']
        self.num_genre = config['num_genre']
        self.num_actor = config['num_actor']
        self.num_director = config['num_director']

        self.embedding_dim = config['embedding_dim']

        self.embedding_itemId = torch.nn.Embedding(
            num_embeddings=self.num_item,
            embedding_dim=self.embedding_dim
        )
        self.embedding_rate = torch.nn.Embedding(
            num_embeddings=self.num_rate,
            embedding_dim=self.embedding_dim
        )

        self.embedding_genre = torch.nn.Linear(
            in_features=self.num_genre,
            out_features=self.embedding_dim,
            bias=False
        ) # torch.nn.Linear()
        self.embedding_actor = torch.nn.Linear(
            in_features=self.num_actor,
            out_features=self.embedding_dim,
            bias=False
        )
        self.embedding_director = torch.nn.Linear(
            in_features=self.num_director,
            out_features=self.embedding_dim,
            bias=False
        )

    def forward(self, item_fea):
        """
        :param item_fea: [ID(0),rate（1），genre（2，2+num_genre），actor(2+num_genre,2+num_genre+num_actors), director(2+num_genre+num_actors: ) ]
        :return:
        """

        itemId_idx = Variable(item_fea[:, 0], requires_grad=False)
        rate_idx = Variable(item_fea[:, 1], requires_grad=False)
        genre_idx = Variable(item_fea[:, 2:2+self.num_genre], requires_grad=False)
        actor_idx = Variable(item_fea[:, 2+self.num_genre : 2+self.num_genre+self.num_actor], requires_grad=False)
        director_idx = Variable(item_fea[:, 2+self.num_genre+self.num_actor:], requires_grad=False)

        itemId_emb = self.embedding_itemId(itemId_idx) # (1,32)
        rate_emb = self.embedding_rate(rate_idx)  # (1,32)

        genre_sum = torch.sum(genre_idx.float(), 1).view(-1, 1)
        genre_emb = self.embedding_genre(genre_idx.float()) / torch.where(genre_sum==0, torch.ones_like(genre_sum) , genre_sum)  # (1,32) view()

        actor_sum = torch.sum(actor_idx.float(), 1).view(-1, 1)
        actor_emb = self.embedding_actor(actor_idx.float()) / torch.where(actor_sum==0, torch.ones_like(actor_sum) , actor_sum)  # (1,32) view()

        director_sum = torch.sum(director_idx.float(), 1).view(-1, 1)
        director_emb = self.embedding_director(director_idx.float()) / torch.where(director_sum==0, torch.ones_like(director_sum) , director_sum)  # (1,32) view() 相当于 reshape
        return torch.cat((itemId_emb,rate_emb, genre_emb,actor_emb,director_emb), 1)  # (samples, 5*32)

class UserEmbeddingML_test(torch.nn.Module):
    def __init__(self, config):
        super(UserEmbeddingML_test, self).__init__()

        self.num_user = config['num_user']
        self.embedding_dim = config['embedding_dim']

        self.embedding_userId = torch.nn.Embedding(
            num_embeddings=self.num_user,
            embedding_dim=self.embedding_dim
        )  # torch.nn.Embedding()

    def forward(self, user_fea):
        """
        :param user_fea: [userId(0),gender(1),age(2),occupation(3),area(4)]
        :return:
        """
        userId_idx = Variable(user_fea[:, 0], requires_grad=False)
        userId_emb = self.embedding_userId(userId_idx)
        return userId_emb

class ItemEmbeddingML_test(torch.nn.Module):
    def __init__(self, config):
        super(ItemEmbeddingML_test, self).__init__()
        self.num_item = config['num_item']
        self.embedding_dim = config['embedding_dim']

        self.embedding_itemId = torch.nn.Embedding(
            num_embeddings=self.num_item,
            embedding_dim=self.embedding_dim
        )

    def forward(self, item_fea):
        """
        :param item_fea: [ID(0),rate（1），genre（2，2+num_genre），actor(2+num_genre,2+num_genre+num_actors), director(2+num_genre+num_actors: ) ]
        :return:
        """

        itemId_idx = Variable(item_fea[:, 0], requires_grad=False)
        itemId_emb = self.embedding_itemId(itemId_idx) # (1,32)
        return itemId_emb  # (samples, 5*32)

class UserEmbeddingDB(torch.nn.Module):
    def __init__(self, config):
        super(UserEmbeddingDB, self).__init__()

        self.num_user = config['num_user']
        self.num_location = config['num_location']

        self.embedding_dim = config['embedding_dim']

        self.embedding_userId = torch.nn.Embedding(
            num_embeddings=self.num_user,
            embedding_dim=self.embedding_dim
        )
        self.embedding_location = torch.nn.Embedding(
            num_embeddings=self.num_location,
            embedding_dim=self.embedding_dim
        )

    def forward(self, user_fea):
        """
        :param user_fea: [userId(0),gender(1),age(2),occupation(3),area(4)]
        :return:
        """
        userId_idx = Variable(user_fea[:, 0], requires_grad=False)
        location_idx = Variable(user_fea[:, 1], requires_grad=False)

        userId_emb = self.embedding_userId(userId_idx)
        location_emb = self.embedding_location(location_idx)
        return torch.cat((userId_emb, location_emb), 1)   # (samples, 2*32)  torch.cat() concatenation

class ItemEmbeddingDB(torch.nn.Module):
    def __init__(self, config):
        super(ItemEmbeddingDB, self).__init__()
        self.num_item = config['num_item']
        self.num_author = config['num_author']
        self.num_publisher = config['num_publisher']
        self.num_year = config['num_year']

        self.embedding_dim = config['embedding_dim']

        self.embedding_itemId = torch.nn.Embedding(
            num_embeddings=self.num_item,
            embedding_dim=self.embedding_dim
        )
        self.embedding_author = torch.nn.Embedding(
            num_embeddings=self.num_author,
            embedding_dim=self.embedding_dim
        )
        self.embedding_publisher = torch.nn.Embedding(
            num_embeddings=self.num_publisher,
            embedding_dim=self.embedding_dim
        )
        self.embedding_year = torch.nn.Embedding(
            num_embeddings=self.num_year,
            embedding_dim=self.embedding_dim
        )

    def forward(self, item_fea):
        """
        :param item_fea: [ID(0),rate（1），genre（2，2+num_genre），actor(2+num_genre,2+num_genre+num_actors), director(2+num_genre+num_actors: ) ]
        :return:
        """

        itemId_idx = Variable(item_fea[:, 0], requires_grad=False)
        author_idx = Variable(item_fea[:, 1], requires_grad=False)
        publisher_idx = Variable(item_fea[:, 2], requires_grad=False)
        year_idx = Variable(item_fea[:, 3], requires_grad=False)

        itemId_emb = self.embedding_itemId(itemId_idx) # (1,32)
        author_emb = self.embedding_author(author_idx)  # (1,32)
        publisher_emb = self.embedding_publisher(publisher_idx)  # (1,32)
        year_emb = self.embedding_year(year_idx)  # (1,32)

        return torch.cat((itemId_emb,author_emb, publisher_emb,year_emb), 1)  # (samples, 5*32)

class UserEmbeddingYP(torch.nn.Module):
    def __init__(self, config):
        super(UserEmbeddingYP, self).__init__()

        self.num_user = config['num_user']
        self.num_fans = config['num_fans']
        self.num_avgrating = config['num_avgrating']

        self.embedding_dim = config['embedding_dim']

        self.embedding_userId = torch.nn.Embedding(
            num_embeddings=self.num_user,
            embedding_dim=self.embedding_dim
        )
        self.embedding_fans = torch.nn.Embedding(
            num_embeddings=self.num_fans,
            embedding_dim=self.embedding_dim
        )
        self.embedding_avgrating = torch.nn.Embedding(
            num_embeddings=self.num_avgrating,
            embedding_dim=self.embedding_dim
        )

    def forward(self, user_fea):
        """
        :param user_fea: [userId(0),gender(1),age(2),occupation(3),area(4)]
        :return:
        """
        userId_idx = Variable(user_fea[:, 0], requires_grad=False)
        fans_idx = Variable(user_fea[:, 1], requires_grad=False)
        avgrating_idx = Variable(user_fea[:, 2], requires_grad=False)

        userId_emb = self.embedding_userId(userId_idx)
        fans_emb = self.embedding_fans(fans_idx)
        avgrating_emb = self.embedding_avgrating(avgrating_idx)
        return torch.cat((userId_emb, fans_emb,avgrating_emb), 1)   # (samples, 3*32)  torch.cat() concatenation

class ItemEmbeddingYP(torch.nn.Module):
    def __init__(self, config):
        super(ItemEmbeddingYP, self).__init__()
        self.num_item = config['num_item']
        self.num_postalcode = config['num_postalcode']
        self.num_stars = config['num_stars']
        self.num_city = config['num_city']
        self.num_category = config['num_category']

        self.embedding_dim = config['embedding_dim']

        self.embedding_itemId = torch.nn.Embedding(
            num_embeddings=self.num_item,
            embedding_dim=self.embedding_dim
        )
        self.embedding_postalcode = torch.nn.Embedding(
            num_embeddings=self.num_postalcode,
            embedding_dim=self.embedding_dim
        )
        self.embedding_stars = torch.nn.Embedding(
            num_embeddings=self.num_stars,
            embedding_dim=self.embedding_dim
        )
        self.embedding_city = torch.nn.Embedding(
            num_embeddings=self.num_city,
            embedding_dim=self.embedding_dim
        )
        self.embedding_category = torch.nn.Linear(
            in_features=self.num_category,
            out_features=self.embedding_dim,
            bias=False
        )

    def forward(self, item_fea):
        """
        :param item_fea: [ID(0),rate（1），genre（2，2+num_genre），actor(2+num_genre,2+num_genre+num_actors), director(2+num_genre+num_actors: ) ]
        :return:
        """

        itemId_idx = Variable(item_fea[:, 0], requires_grad=False)
        postalcode_idx = Variable(item_fea[:, 1], requires_grad=False)
        stars_idx = Variable(item_fea[:, 2], requires_grad=False)
        city_idx = Variable(item_fea[:, 3], requires_grad=False)
        category_idx = Variable(item_fea[:, 4:], requires_grad=False)

        itemId_emb = self.embedding_itemId(itemId_idx) # (1,32)
        postalcode_emb = self.embedding_postalcode(postalcode_idx)  # (1,32)
        stars_emb = self.embedding_stars(stars_idx)  # (1,32)
        city_emb = self.embedding_city(city_idx)  # (1,32)

        category_sum = torch.sum(category_idx.float(), 1).view(-1, 1)
        category_emb = self.embedding_category(category_idx.float()) / torch.where(category_sum == 0, torch.ones_like(category_sum),
                                                                          category_sum)

        return torch.cat((itemId_emb,postalcode_emb, stars_emb,city_emb,category_emb), 1)  # (samples, 5*32)

class NCF_RecommModule(torch.nn.Module):
    def __init__(self,config):
        super(NCF_RecommModule, self).__init__()
        self.config = config
        self.embedding_dim = config['embedding_dim']
        self.fc1_in_dim = config['user_embedding_dim'] + config['item_embedding_dim']
        self.fc2_in_dim = config['first_fc_hidden_dim']
        self.fc2_out_dim = config['second_fc_hidden_dim']
        self.use_cuda = config['use_cuda']


        self.vars = torch.nn.ParameterDict()
        # self.vars_bn = torch.nn.ParameterList()


        w1 = torch.nn.Parameter(torch.ones([self.fc2_in_dim,self.fc1_in_dim]))  # layer_1
        torch.nn.init.xavier_normal_(w1)
        self.vars['recomm_fc_w1'] = w1
        self.vars['recomm_fc_b1'] = torch.nn.Parameter(torch.zeros(self.fc2_in_dim))

        w2 = torch.nn.Parameter(torch.ones([self.fc2_out_dim,self.fc2_in_dim])) # layer_2
        torch.nn.init.xavier_normal_(w2)
        self.vars['recomm_fc_w2'] = w2
        self.vars['recomm_fc_b2'] = torch.nn.Parameter(torch.zeros(self.fc2_in_dim))

        w3 = torch.nn.Parameter(torch.ones([1, self.fc2_out_dim])) # output_layer
        torch.nn.init.xavier_normal_(w3)
        self.vars['recomm_fc_w3'] = w3
        self.vars['recomm_fc_b3'] = torch.nn.Parameter(torch.zeros(1))

    def forward(self, user_emb, item_emb, vars_dict=None):
        """
        """
        if vars_dict is None:
            vars_dict = self.vars

        x_i = item_emb
        x_u = user_emb  # movielens: loss:12.14... up! ; dbook 20epoch: user_cold: mae 0.6051;

        x = torch.cat((x_i, x_u), 1)
        x = F.relu(F.linear(x, vars_dict['recomm_fc_w1'], vars_dict['recomm_fc_b1']))
        x = F.relu(F.linear(x, vars_dict['recomm_fc_w2'], vars_dict['recomm_fc_b2']))
        x = F.linear(x, vars_dict['recomm_fc_w3'], vars_dict['recomm_fc_b3'])

        return x.squeeze()

    def zero_grad(self, vars_dict=None):
        with torch.no_grad():
            if vars_dict is None:
                for p in self.vars.values():
                    if p.grad is not None:
                        p.grad.zero_()
            else:
                for p in vars_dict.values():
                    if p.grad is not None:
                        p.grad.zero_()

    def update_parameters(self):
        return self.vars

