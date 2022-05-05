# https://github.com/XWilliamY/custom_yt_comments_dataset/blob/master/get_comments_of_video_id.py
# https://towardsdatascience.com/how-to-build-your-own-dataset-of-youtube-comments-39a1e57aade

import csv
from openpyxl import Workbook
import pandas as pd
import argparse
from googleapiclient.discovery import build
from urllib.parse import urlparse, parse_qs

# https://stackoverflow.com/questions/45579306/get-youtube-video-url-or-youtube-video-id-from-a-string-using-regex
def get_id(url):
    url_parse = urlparse(url)
    quer_v = parse_qs(url_parse.query).get('v')
    if quer_v:
        return quer_v[0]
    pth = url_parse.path.split('/')
    if pth:
        return pth[-1]

def build_service():
    # with open(filename) as f:
    #     key = f.readline()

    api_key = 'AIzaSyD_NXS0Qsr1kTNAjz1Vk9DMZyRq1IeFzA0'
    api_service_name = "youtube"
    api_version = "v3"
    return build(api_service_name, api_version, developerKey=api_key)

# def get_comments(part='snippet', 
#                  maxResults=100, 
#                  textFormat='plainText',
#                  order='time',
#                  videoId='9bqk6ZUsKyA',
#                  csv_filename="mrbeast"):

def get_comments(**kwargs):

    # create empty lists to store desired information
    comments, commentsId, repliesCount, likesCount, publishedAt = [], [], [], [], []
       
    kwargs['part'] = kwargs.get('part', 'snippet').split()
    kwargs['maxResults'] = kwargs.get('maxResults', 100)
    kwargs['textFormat'] = kwargs.get('textFormat', 'plainText')
    kwargs['order'] = kwargs.get('order', 'time')
    service = kwargs.pop('service')

    write_lbl = kwargs.pop('write_lbl', True)
    csv_filename = kwargs.pop('csv_filename')
    token_filename = kwargs.pop('token_filename')
    
    # make an API call using our service
    response = service.commentThreads().list(
        **kwargs
    ).execute()
                 
    page = 0
    while response: # this loop will continue to run until you max out your quota
        print(f'page {page}')
        page += 1
        index = 0
        for item in response['items']:
            index += 1

            # index item for desired data features
            comment = item['snippet']['topLevelComment']['snippet']['textDisplay']
            comment_id = item['snippet']['topLevelComment']['id']
            reply_count = item['snippet']['totalReplyCount']
            like_count = item['snippet']['topLevelComment']['snippet']['likeCount']
            published_at = item['snippet']['topLevelComment']['snippet']['publishedAt']
            
            # append to lists
            comments.append(comment)
            commentsId.append(comment_id)
            repliesCount.append(reply_count)
            likesCount.append(like_count)
            publishedAt.append(published_at)

            # write line by line
            if write_lbl:
                with open(f'{csv_filename}.csv', 'a+', encoding="utf-8") as f:
                    csv_writer = csv.writer(f)
                    csv_writer.writerow([comment, comment_id, reply_count, like_count])
        
        # https://thispointer.com/python-how-to-append-a-new-row-to-an-existing-csv-file/
        
        # check for nextPageToken, and if it exists, set response equal to the JSON response
        if 'nextPageToken' in response:
            with open(f'{token_filename}.txt', 'a+') as f:
                f.write(kwargs.get('pageToken', ''))
                f.write('\n')
            kwargs['pageToken'] = response['nextPageToken']
            response = service.commentThreads().list(
                **kwargs
            ).execute()
        else:
            break

    return {
        'Comments': comments,
        'Comment ID': commentsId,
        'Reply count': repliesCount,
        'Like count': likesCount,
        'Published at': publishedAt
    }

def save_to_csv(output_dict, video_id, output_filename):
    output_df = pd.DataFrame(output_dict, columns = output_dict.keys())
    output_df.to_csv(f'{output_filename}.csv')

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--part', help='Desired properties of commentThread', default='snippet')
    parser.add_argument('--maxResults', help='Max results per page', default=100)
    parser.add_argument('--write_lbl', help="Update csv after each comment?", default=True)
    parser.add_argument('--csv_filename', default=None)
    parser.add_argument('--token_filename', default=None)
    parser.add_argument('--video_url', default='https://www.youtube.com/watch?v=4uEC_xibqNY')
    parser.add_argument('--order', default='time')
    parser.add_argument('--pageToken', default=None)
    args = parser.parse_args()

    # build kwargs from args
    kwargs = vars(args)

    service = build_service()
    video_id = get_id(kwargs.pop('video_url'))

    if not args.csv_filename:
        args.csv_filename = video_id + "_csv"

    if not args.token_filename:
        args.token_filename = video_id + "_page_token"

    if not kwargs.get('pageToken'):
        kwargs.pop('pageToken')

    kwargs['videoId'] = video_id
    kwargs['service'] = service
    output_dict = get_comments(**kwargs)

    args.csv_filename += "_final"
    save_to_csv(output_dict, video_id, args.csv_filename)

    #save csv as excel since csv messes up many non-English characters
    wb = Workbook()
    ws = wb.active
    with open(f'{args.csv_filename}.csv', 'r', encoding='utf-8') as f:
        for row in csv.reader(f):
            ws.append(row)
    wb.save(f'{args.csv_filename}.xlsx')
    
if __name__ == '__main__':
    main()