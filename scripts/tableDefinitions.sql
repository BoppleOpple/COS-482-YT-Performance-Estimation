CREATE TABLE videos (
	id varchar(20) PRIMARY KEY,
	title varchar(100),
	description varchar(5000),
	thumbnail varchar(200),
	duration integer,
	posted_time timestamp
);

CREATE TABLE channels (
	id varchar(50) PRIMARY KEY,
	name varchar(100),
	views bigint,
	subscribers integer,
	videos integer
);

CREATE TABLE categories (
	id integer PRIMARY KEY,
	name varchar(200)
);

CREATE TABLE queries (
	category_id integer,
	time timestamp,
	CONSTRAINT fk_category_id FOREIGN KEY (category_id) REFERENCES categories(id) DEFERRABLE INITIALLY DEFERRED,
	PRIMARY KEY (category_id, time)
);

CREATE TABLE tags (
	tag varchar(300) PRIMARY KEY
);

CREATE TABLE created_by (
	video_id varchar(20) REFERENCES videos(id),
	channel_id varchar(50) REFERENCES channels(id),
	CONSTRAINT fk_video_id FOREIGN KEY (video_id) REFERENCES videos(id) DEFERRABLE INITIALLY DEFERRED,
	CONSTRAINT fk_channel_id FOREIGN KEY (channel_id) REFERENCES channels(id) DEFERRABLE INITIALLY DEFERRED,
	PRIMARY KEY (video_id, channel_id)
);

CREATE TABLE in_category (
	video_id varchar(20),
	category_id integer,
	CONSTRAINT fk_video_id FOREIGN KEY (video_id) REFERENCES videos(id) DEFERRABLE INITIALLY DEFERRED,
	CONSTRAINT fk_category_id FOREIGN KEY (category_id) REFERENCES categories(id) DEFERRABLE INITIALLY DEFERRED,
	PRIMARY KEY (video_id, category_id)
);

CREATE TABLE access_information (
	video_id varchar(20),
	query_category integer,
	query_time timestamp,
	views integer,
	likes integer,
	comments integer,
	CONSTRAINT fk_video_id FOREIGN KEY (video_id) REFERENCES videos(id) DEFERRABLE INITIALLY DEFERRED,
	CONSTRAINT fk_query FOREIGN KEY (query_category, query_time) REFERENCES queries(category_id, time) DEFERRABLE INITIALLY DEFERRED,
	PRIMARY KEY (video_id, query_category, query_time)
);

CREATE INDEX idx_vid ON access_information(video_id);

CREATE TABLE has_tag (
	video_id varchar(20),
	tag varchar(300),
	CONSTRAINT fk_video_id FOREIGN KEY (video_id) REFERENCES videos(id) DEFERRABLE INITIALLY DEFERRED,
	CONSTRAINT fk_tag FOREIGN KEY (tag) REFERENCES tags(tag) DEFERRABLE INITIALLY DEFERRED,
	PRIMARY KEY (video_id, tag)
);